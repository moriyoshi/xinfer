use super::{DensePackedKvPage, PackedKvAxis};
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::Mutex;

const MODULE: &str = "xinfer_dense_packed_kv_encode_v1";
const PARAM_KERNEL: &str = "dense_kv_params";
const CODE_KERNEL: &str = "dense_kv_codes";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

fn as_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("dense packed KV encoder geometry exceeds u32".into()))
}

pub(super) fn encode(
    tile: &Tensor,
    tokens: usize,
    axis: PackedKvAxis,
    bits: u8,
    exact_tail_tokens: usize,
) -> Result<Vec<DensePackedKvPage>> {
    const BLOCK: usize = 16;
    if !matches!(bits, 2 | 4) || tile.dtype() != DType::BF16 || tile.rank() != 4 {
        candle_core::bail!("GPU packed KV encoding requires rank-four BF16 Flash slots")
    }
    let (blocks, block_tokens, heads, channels) = tile.dims4()?;
    let slots = blocks
        .checked_mul(block_tokens)
        .ok_or_else(|| candle_core::Error::Msg("KV slot count overflow".into()))?;
    let width = heads
        .checked_mul(channels)
        .ok_or_else(|| candle_core::Error::Msg("KV width overflow".into()))?;
    if tokens == 0 || tokens > slots || heads == 0 || channels == 0 {
        candle_core::bail!("invalid GPU packed KV prefix geometry")
    }
    let pages = tokens.div_ceil(BLOCK);
    let exact_from = tokens.saturating_sub(exact_tail_tokens);
    let key = usize::from(axis == PackedKvAxis::Key);
    let max_groups = if key == 1 {
        width
    } else {
        BLOCK
            .checked_mul(heads)
            .ok_or_else(|| candle_core::Error::Msg("KV parameter count overflow".into()))?
    };
    let param_stride = max_groups
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("KV parameter count overflow".into()))?;
    let code_stride = BLOCK
        .checked_mul(width)
        .and_then(|n| n.checked_mul(bits as usize))
        .ok_or_else(|| candle_core::Error::Msg("KV code count overflow".into()))?
        .div_ceil(8);
    let source_words = slots
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("KV tile size overflow".into()))?;
    let param_words = pages
        .checked_mul(param_stride)
        .ok_or_else(|| candle_core::Error::Msg("KV parameter batch overflow".into()))?;
    let code_bytes = pages
        .checked_mul(code_stride)
        .ok_or_else(|| candle_core::Error::Msg("KV code batch overflow".into()))?;
    for value in [source_words, param_words, code_bytes, pages, width] {
        as_u32(value)?;
    }
    let (storage, layout) = tile.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("GPU packed KV encoding requires CUDA storage")
    };
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("noncontiguous GPU KV tile".into()))?;
    if end - start != source_words {
        candle_core::bail!("invalid GPU KV tile extent")
    }
    let source = storage.as_cuda_slice::<half::bf16>()?.slice(start..end);
    let device = storage.device.cuda_device();
    let capture = capture_status(*device.cu_stream())
        .map_err(|e| candle_core::Error::Msg(format!("check CUDA graph capture: {e}")))?;
    if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        candle_core::bail!("GPU packed KV encoding cannot run during CUDA graph capture")
    }
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !device.has_func(MODULE, PARAM_KERNEL) {
            let ptx = compile_ptx(include_str!("encode.cu"))
                .map_err(|e| candle_core::Error::Msg(format!("compile packed KV encoder: {e}")))?;
            device
                .load_ptx(ptx, MODULE, &[PARAM_KERNEL, CODE_KERNEL])
                .map_err(|e| candle_core::Error::Msg(format!("load packed KV encoder: {e}")))?;
        }
    }
    let d_params = device
        .alloc_zeros::<f32>(param_words, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU packed KV parameters: {e}")))?;
    let d_codes = device
        .alloc_zeros::<u8>(code_bytes, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU packed KV codes: {e}")))?;
    let d_invalid = device
        .alloc_zeros::<u32>(1, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU packed KV status: {e}")))?;
    let param_launch = LaunchConfig {
        grid_dim: (as_u32(max_groups.div_ceil(128))?, as_u32(pages)?, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let code_launch = LaunchConfig {
        grid_dim: (as_u32(code_stride.div_ceil(256))?, as_u32(pages)?, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let params_fn = device
        .get_func(MODULE, PARAM_KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing GPU KV parameter kernel".into()))?;
    // Buffers and source view live through synchronous readback below.
    unsafe {
        params_fn.launch(
            param_launch,
            (
                &source,
                &d_params,
                &d_invalid,
                as_u32(tokens)?,
                as_u32(heads)?,
                as_u32(channels)?,
                as_u32(exact_from)?,
                bits as u32,
                key as u32,
                as_u32(param_stride)?,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch GPU KV parameters: {e}")))?;
    let codes_fn = device
        .get_func(MODULE, CODE_KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing GPU KV code kernel".into()))?;
    unsafe {
        codes_fn.launch(
            code_launch,
            (
                &source,
                &d_params,
                &d_codes,
                as_u32(tokens)?,
                as_u32(heads)?,
                as_u32(channels)?,
                as_u32(exact_from)?,
                bits as u32,
                key as u32,
                as_u32(param_stride)?,
                as_u32(code_stride)?,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch GPU KV codes: {e}")))?;
    let invalid = device
        .dtoh_sync_copy(&d_invalid)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV status: {e}")))?;
    if invalid[0] != 0 {
        candle_core::bail!("non-finite BF16 KV value")
    }
    let params = device
        .dtoh_sync_copy(&d_params)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV parameters: {e}")))?;
    let codes = device
        .dtoh_sync_copy(&d_codes)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV codes: {e}")))?;
    let tail_words = tokens - exact_from;
    let tail = if tail_words == 0 {
        Vec::new()
    } else {
        tile.reshape((slots, width))?
            .narrow(0, exact_from, tail_words)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?
    };
    if tail.iter().any(|word| !word.to_f32().is_finite()) {
        candle_core::bail!("non-finite BF16 KV value")
    }
    let mut output = Vec::with_capacity(pages);
    for page in 0..pages {
        let first = page * BLOCK;
        let count = (tokens - first).min(BLOCK);
        let old = count.min(exact_from.saturating_sub(first));
        let tail_count = count - old;
        let code_len = (old * width * bits as usize).div_ceil(8);
        let group_count = if key == 1 && old > 0 {
            width
        } else {
            old * heads
        };
        let mut exact = Vec::with_capacity(tail_count * width);
        if key == 1 {
            for hd in 0..width {
                for token in old..count {
                    exact.push(tail[(first + token - exact_from) * width + hd].to_bits());
                }
            }
        } else {
            for token in old..count {
                let offset = (first + token - exact_from) * width;
                exact.extend(
                    tail[offset..offset + width]
                        .iter()
                        .map(|word| word.to_bits()),
                );
            }
        }
        output.push(DensePackedKvPage::new(
            axis,
            bits,
            count as u32,
            as_u32(heads)?,
            as_u32(channels)?,
            if key == 1 { 16 } else { as_u32(channels)? },
            as_u32(tail_count)?,
            codes[page * code_stride..page * code_stride + code_len].to_vec(),
            params[page * param_stride..page * param_stride + group_count * 2].to_vec(),
            exact,
            Vec::new(),
        )?);
    }
    Ok(output)
}
