//! Experimental wider-tile paged BF16 prefill for Nemotron-H on Blackwell.
//! Cache writes use the regular attention-rs path; only prefill QK/PV changes.

use attention_rs::InputMetadata;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Storage, Tensor};

extern "C" {
    fn xinfer_nemotron_flash_prefill_bf16(
        q: *const std::ffi::c_void,
        k_cache: *const std::ffi::c_void,
        v_cache: *const std::ffi::c_void,
        k_direct: *const std::ffi::c_void,
        v_direct: *const std::ffi::c_void,
        direct_current: bool,
        output: *mut std::ffi::c_void,
        block_tables: *const i32,
        block_table_stride: u32,
        cu_seqlens_q: *const u32,
        context_lens: *const u32,
        num_seqs: u32,
        max_seqlen_q: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        cache_block_size: u32,
        scale: f32,
        stream: i64,
    ) -> i32;
}

fn valid_current_lengths(metadata: &InputMetadata, sequences: usize, rows: usize) -> bool {
    let (Some(contexts), Some(cumulative)) = (
        metadata.context_lens_host.as_ref(),
        metadata.seqlens.as_ref(),
    ) else {
        return false;
    };
    if contexts.len() != sequences || cumulative.len() != sequences {
        return false;
    }
    let mut previous = 0usize;
    for (&context, &end) in contexts.iter().zip(cumulative.iter()) {
        let end = end as usize;
        if end <= previous || (context as usize) < end - previous {
            return false;
        }
        previous = end;
    }
    previous == rows
}

fn ptr(t: &Tensor, dtype: DType) -> Result<u64> {
    if t.dtype() != dtype || !t.is_contiguous() {
        candle_core::bail!("Nemotron attention requires contiguous {dtype:?} tensor")
    }
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Nemotron attention requires CUDA tensors")
    };
    let base = match dtype {
        DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
        DType::U32 => *storage.as_cuda_slice::<u32>()?.device_ptr(),
        _ => candle_core::bail!("unsupported Nemotron attention dtype {dtype:?}"),
    };
    Ok(base + (layout.start_offset() * dtype.size_in_bytes()) as u64)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_prefill(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cache: Option<(&Tensor, &Tensor)>,
    metadata: &InputMetadata,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Option<Tensor>> {
    let Some((key_cache, value_cache)) = cache else {
        return Ok(None);
    };
    let (Some(block_tables), Some(context_lens), Some(cu_seqlens_q)) = (
        metadata.block_tables.as_ref(),
        metadata.context_lens.as_ref(),
        metadata.cu_seqlens_q.as_ref(),
    ) else {
        return Ok(None);
    };
    if !metadata.is_prefill
        || query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || key_cache.dtype() != DType::BF16
        || value_cache.dtype() != DType::BF16
        || head_dim != 128
        || q_heads != 32
        || kv_heads != 2
        || !query.is_contiguous()
        || !key.is_contiguous()
        || !value.is_contiguous()
    {
        return Ok(None);
    }
    let dev = query.device().as_cuda_device()?;
    if attention_rs::cuda_utils::sm_version(dev).unwrap_or(0) < 120 {
        return Ok(None);
    }
    let (rows, qh, width) = query.dims3()?;
    let block_size = key_cache.dim(1)?;
    let (num_sequences, table_stride) = block_tables.dims2()?;
    if !block_size.is_power_of_two() {
        return Ok(None);
    }
    if qh != q_heads
        || width != head_dim
        || key.dims() != [rows, kv_heads, head_dim]
        || value.dims() != [rows, kv_heads, head_dim]
        || key_cache.dims() != value_cache.dims()
        || key_cache.dim(2)? != kv_heads
        || key_cache.dim(3)? != head_dim
        || metadata.slot_mapping.elem_count() != rows
        || context_lens.elem_count() != num_sequences
        || cu_seqlens_q.elem_count() != num_sequences + 1
        || metadata.max_seqlen_q > rows
        || metadata.max_context_len > table_stride * block_size
    {
        candle_core::bail!("Nemotron attention tensor shape mismatch")
    }
    if num_sequences == 0 || metadata.max_seqlen_q == 0 {
        candle_core::bail!("Nemotron attention requires nonempty prefill metadata")
    }
    let slots = metadata.slot_mapping.flatten_all()?;
    attention_rs::flash::flash_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        None,
        None,
        &slots,
    )?;
    // The grid covers every (sequence, query tile, query head), and each
    // thread writes disjoint BF16 pairs for all valid rows and channels.
    let output = unsafe { Tensor::empty_(query.shape(), DType::BF16, query.device())? };
    // Prefix rows already in the cache keep their paged lookup. Rows in this
    // prefill call are contiguous and can skip the page-table lookup. Missing
    // host lengths conservatively retain the paged path.
    let direct_kv = std::env::var_os("XINFER_NEMOTRON_DIRECT_PREFILL_KV").as_deref()
        != Some(std::ffi::OsStr::new("0"))
        && valid_current_lengths(metadata, num_sequences, rows);
    let stream = *dev.cu_stream() as i64;
    let result = unsafe {
        xinfer_nemotron_flash_prefill_bf16(
            ptr(query, DType::BF16)? as *const _,
            ptr(key_cache, DType::BF16)? as *const _,
            ptr(value_cache, DType::BF16)? as *const _,
            ptr(key, DType::BF16)? as *const _,
            ptr(value, DType::BF16)? as *const _,
            direct_kv,
            ptr(&output, DType::BF16)? as *mut _,
            ptr(block_tables, DType::U32)? as *const i32,
            table_stride as u32,
            ptr(cu_seqlens_q, DType::U32)? as *const u32,
            ptr(context_lens, DType::U32)? as *const u32,
            num_sequences as u32,
            metadata.max_seqlen_q as u32,
            q_heads as u32,
            kv_heads as u32,
            block_size as u32,
            scale,
            stream,
        )
    };
    if result != 0 {
        candle_core::bail!("Nemotron attention CUDA launch failed: {result}")
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn query_tile_matches_paged_flash_context() -> anyhow::Result<()> {
        if std::env::var_os("XINFER_NEMOTRON_ATTN_CUDA_TEST").is_none() {
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let rows = std::env::var("XINFER_NEMOTRON_ATTN_ROWS")
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(129usize);
        let block_size = 16usize;
        let sequence_lengths = if std::env::var_os("XINFER_NEMOTRON_ATTN_BATCH").is_some() {
            vec![rows / 2, rows - rows / 2]
        } else {
            vec![rows]
        };
        let prefix = std::env::var("XINFER_NEMOTRON_ATTN_PREFIX")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let table_stride = sequence_lengths
            .iter()
            .map(|length| (length + prefix).div_ceil(block_size))
            .max()
            .unwrap();
        let pages = sequence_lengths
            .iter()
            .map(|length| (length + prefix).div_ceil(block_size))
            .sum::<usize>();
        let mut slots_host = Vec::with_capacity(rows);
        let mut prefix_slots_host = Vec::with_capacity(prefix * sequence_lengths.len());
        let mut tables_host = vec![0u32; sequence_lengths.len() * table_stride];
        let mut cu_host = vec![0u32];
        let mut page_base = 0usize;
        for (sequence, &length) in sequence_lengths.iter().enumerate() {
            let page_count = (length + prefix).div_ceil(block_size);
            for page in 0..page_count {
                tables_host[sequence * table_stride + page] = (page_base + page) as u32;
            }
            for position in 0..length + prefix {
                let slot = ((page_base + position / block_size) * block_size
                    + position % block_size) as i64;
                if position < prefix {
                    prefix_slots_host.push(slot);
                } else {
                    slots_host.push(slot);
                }
            }
            page_base += page_count;
            cu_host.push(cu_host.last().unwrap() + length as u32);
        }
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| (((i * 113 + seed) % 307) as f32 - 153.0) / 64.0)
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_vec(values(rows * 32 * 128, 7), (rows, 32, 128), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(values(rows * 2 * 128, 37), (rows, 2, 128), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::from_vec(values(rows * 2 * 128, 91), (rows, 2, 128), &device)?
            .to_dtype(DType::BF16)?;
        let kc = Tensor::zeros((pages, block_size, 2, 128), DType::BF16, &device)?;
        let vc = Tensor::zeros((pages, block_size, 2, 128), DType::BF16, &device)?;
        if prefix > 0 {
            let prefix_rows = prefix * sequence_lengths.len();
            let prefix_k = Tensor::from_vec(
                values(prefix_rows * 2 * 128, 17),
                (prefix_rows, 2, 128),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let prefix_v = Tensor::from_vec(
                values(prefix_rows * 2 * 128, 53),
                (prefix_rows, 2, 128),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let prefix_slots = Tensor::from_vec(prefix_slots_host, prefix_rows, &device)?;
            attention_rs::flash::flash_reshape_and_cache(
                &prefix_k,
                &prefix_v,
                &kc,
                &vc,
                None,
                None,
                &prefix_slots,
            )?;
        }
        let slots = Tensor::from_vec(slots_host, rows, &device)?;
        let blocks =
            Tensor::from_vec(tables_host, (sequence_lengths.len(), table_stride), &device)?;
        let contexts = Tensor::from_vec(
            sequence_lengths
                .iter()
                .map(|length| (length + prefix) as u32)
                .collect(),
            sequence_lengths.len(),
            &device,
        )?;
        let cu = Tensor::from_vec(cu_host.clone(), sequence_lengths.len() + 1, &device)?;
        let mut metadata = InputMetadata {
            is_prefill: true,
            is_mla: false,
            sequence_ids: None,
            mamba_slot_mapping: None,
            slot_mapping: slots,
            block_tables: Some(blocks.clone()),
            block_tables_host: None,
            context_lens_host: Some(
                sequence_lengths
                    .iter()
                    .map(|&n| (n + prefix) as u32)
                    .collect(),
            ),
            context_lens: Some(contexts.clone()),
            cu_seqlens_q: Some(cu.clone()),
            cu_seqlens_k: None,
            max_seqlen_q: *sequence_lengths.iter().max().unwrap(),
            max_seqlen_k: sequence_lengths.iter().max().unwrap() + prefix,
            max_context_len: sequence_lengths.iter().max().unwrap() + prefix,
            seqlens: Some(cu_host[1..].to_vec()),
            flashinfer_metadata: None,
            is_mtp_verify: false,
        };
        let actual = try_prefill(
            &q,
            &k,
            &v,
            Some((&kc, &vc)),
            &metadata,
            32,
            2,
            128,
            1.0 / (128.0f32).sqrt(),
        )?
        .expect("GB10 specialization");
        let expected = attention_rs::flash::flash_prefill(
            &q,
            &kc,
            &vc,
            &blocks,
            &contexts,
            32,
            2,
            128,
            1.0 / (128.0f32).sqrt(),
            0.0,
            None,
            None,
            None,
            Some(&cu),
            metadata.max_seqlen_q,
        )?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("query_tile_context_max_abs={max_abs}");
        anyhow::ensure!(
            max_abs == 0.0,
            "attention context differs from paged BF16 oracle: {max_abs}"
        );
        if std::env::var_os("XINFER_NEMOTRON_ATTN_BENCH").is_some() {
            use std::time::Instant;
            let mut baseline = Vec::new();
            let mut custom = Vec::new();
            for _ in 0..6 {
                let began = Instant::now();
                attention_rs::flash::flash_reshape_and_cache(
                    &k,
                    &v,
                    &kc,
                    &vc,
                    None,
                    None,
                    &metadata.slot_mapping,
                )?;
                let _ = attention_rs::flash::flash_prefill(
                    &q,
                    &kc,
                    &vc,
                    &blocks,
                    &contexts,
                    32,
                    2,
                    128,
                    1.0 / (128.0f32).sqrt(),
                    0.0,
                    None,
                    None,
                    None,
                    Some(&cu),
                    metadata.max_seqlen_q,
                )?;
                device.synchronize()?;
                baseline.push(began.elapsed().as_secs_f64() * 1000.0);
                let began = Instant::now();
                let _ = try_prefill(
                    &q,
                    &k,
                    &v,
                    Some((&kc, &vc)),
                    &metadata,
                    32,
                    2,
                    128,
                    1.0 / (128.0f32).sqrt(),
                )?;
                device.synchronize()?;
                custom.push(began.elapsed().as_secs_f64() * 1000.0);
            }
            eprintln!("attention_prefill_ms_baseline={baseline:?} custom={custom:?}");
        }
        assert!(valid_current_lengths(
            &metadata,
            sequence_lengths.len(),
            rows
        ));
        metadata.context_lens_host.as_mut().unwrap()[0] += 1;
        assert!(valid_current_lengths(
            &metadata,
            sequence_lengths.len(),
            rows
        ));
        metadata.context_lens_host.as_mut().unwrap()[0] = 0;
        assert!(!valid_current_lengths(
            &metadata,
            sequence_lengths.len(),
            rows
        ));
        Ok(())
    }
}
