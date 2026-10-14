//! Xinfer-owned fused expert gather/activation quantization for Nemotron-H.
//! Based on the pinned attention-rs grouped NVFP4 hardware wrapper (MIT).
//! The only changed stage is activation quantization, which reads routed rows
//! directly from their source instead of materializing a BF16 gather buffer.

use attention_rs::workspace::{get_moe_activation_pool, get_moe_cutlass_workspace};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Result, Tensor};
use kernels::ffi;

fn pad_to(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

unsafe extern "C" {
    fn xinfer_nvfp4_quantize_gathered_bf16(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        scales: *mut std::ffi::c_void,
        sorted_ids: *const i32,
        map_divisor: i32,
        input_scale_invs: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        rows: i32,
        experts: i32,
        k: i32,
        k_scale_padded: i32,
        stream: i64,
    );
    fn xinfer_nvfp4_quantize_gathered_f16(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        scales: *mut std::ffi::c_void,
        sorted_ids: *const i32,
        map_divisor: i32,
        input_scale_invs: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        rows: i32,
        experts: i32,
        k: i32,
        k_scale_padded: i32,
        stream: i64,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn moe_gemm_nvfp4_fused_quant(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
    input_scales: Option<&Tensor>,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    _is_prefill: bool,
    pre_swizzled_weight_scales: Option<&Tensor>,
) -> Result<Tensor> {
    use candle_core::{DType, Storage};

    fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = if weights.is_contiguous() {
        weights.clone()
    } else {
        weights.contiguous()?
    };
    let weight_scales = if weight_scales.is_contiguous() {
        weight_scales.clone()
    } else {
        weight_scales.contiguous()?
    };
    let weight_global_scales = if weight_global_scales.is_contiguous() {
        weight_global_scales.clone()
    } else {
        weight_global_scales.contiguous()?
    };
    let sorted_token_ids = if sorted_token_ids.is_contiguous() {
        sorted_token_ids.clone()
    } else {
        sorted_token_ids.contiguous()?
    };
    let experts_ids = if experts_ids.is_contiguous() {
        experts_ids.clone()
    } else {
        experts_ids.contiguous()?
    };

    let (input_rows, k) = input.dims2()?;
    let (num_experts, n, packed_k) = weights.dims3()?;
    if packed_k != k / 2 {
        candle_core::bail!(
            "moe_gemm_nvfp4: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            weights.dims()
        );
    }
    if k % 16 != 0 {
        candle_core::bail!("moe_gemm_nvfp4: K must be divisible by {}, got K={k}", 16);
    }

    let size_m = sorted_token_ids.elem_count();
    if experts_ids.elem_count() != size_m {
        candle_core::bail!(
            "moe_gemm_nvfp4: routed tensors must have {} elements, got sorted={} experts={}",
            size_m,
            sorted_token_ids.elem_count(),
            experts_ids.elem_count()
        );
    }
    let map_divisor = if input_rows == size_m {
        1
    } else if input_rows * topk == size_m {
        topk as i32
    } else {
        candle_core::bail!(
            "moe_gemm_nvfp4_hardware: input rows {} are incompatible with routed size {} and topk {}",
            input_rows,
            size_m,
            topk
        );
    };

    let dtype = input.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) {
        candle_core::bail!("moe_gemm_nvfp4_hardware only accepts f16/bf16 inputs");
    }

    let dev = input.device();
    let cuda_dev = dev.as_cuda_device()?;
    let sm = attention_rs::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
    if sm < 100 {
        candle_core::bail!(
            "moe_gemm_nvfp4_hardware requires Blackwell (sm100+), got sm{}",
            sm
        );
    }

    let stream = *cuda_dev.cu_stream() as i64;
    let k_scale = k / 16;
    let k_scale_padded = pad_to(k_scale, 4);
    let n_padded = pad_to(n, 128);
    let total_sf_rows_capacity = size_m + 127 * num_experts;
    let dtype_bytes = dtype.size_in_bytes();

    let gathered_bytes = 1;
    let act_packed_bytes = size_m * (k / 2);
    let act_scales_bytes = total_sf_rows_capacity * k_scale_padded;
    // Keep the grouped GEMM result in FP32.  The hardware path applies the
    // per-expert alpha in CUTLASS and the top-k route weight during scatter;
    // storing either intermediate in F16/BF16 introduces a second rounding
    // step and diverges from the FP32 reference for valid NVFP4 results.
    let rep_out_bytes = size_m * n * std::mem::size_of::<f32>();
    let output_bytes = size_m * n * dtype_bytes;
    let metadata_bytes = 1;

    let pool = get_moe_activation_pool(
        cuda_dev,
        gathered_bytes,
        act_packed_bytes,
        act_scales_bytes,
        rep_out_bytes,
        output_bytes,
        metadata_bytes,
    )?;

    let expert_counts_t = Tensor::zeros((num_experts,), DType::U32, dev)?;
    let expert_offsets_t = Tensor::zeros((num_experts + 1,), DType::U32, dev)?;
    let sf_offsets_t = Tensor::zeros((num_experts,), DType::U32, dev)?;
    let problem_sizes_t = Tensor::zeros((num_experts * 3,), DType::U32, dev)?;
    let alphas_t = Tensor::zeros((num_experts,), DType::F32, dev)?;
    let input_scale_invs_t = Tensor::zeros((num_experts,), DType::F32, dev)?;

    {
        let (experts_ids_s, _) = experts_ids.storage_and_layout();
        let (expert_counts_s, _) = expert_counts_t.storage_and_layout();
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        unsafe {
            ffi::calculate_expert_offsets(
                cuda_ptr(&experts_ids_s, DType::U32)? as *const i32,
                cuda_ptr(&expert_counts_s, DType::U32)? as *mut i32,
                cuda_ptr(&expert_offsets_s, DType::U32)? as *mut i32,
                num_experts as i32,
                size_m as i32,
                stream,
            );
        }
    }

    {
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        let (weight_global_scales_s, _) = weight_global_scales.storage_and_layout();
        let (sf_offsets_s, _) = sf_offsets_t.storage_and_layout();
        let (problem_sizes_s, _) = problem_sizes_t.storage_and_layout();
        let (alphas_s, _) = alphas_t.storage_and_layout();
        let (input_scale_invs_s, _) = input_scale_invs_t.storage_and_layout();
        let input_scales_ffi_ptr = if let Some(scales) = input_scales {
            let (scales_s, _) = scales.storage_and_layout();
            cuda_ptr(&scales_s, DType::F32)? as *const f32
        } else {
            std::ptr::null()
        };
        unsafe {
            ffi::nvfp4_moe_build_metadata(
                cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                cuda_ptr(&weight_global_scales_s, DType::F32)? as *const f32,
                input_scales_ffi_ptr,
                cuda_ptr(&sf_offsets_s, DType::U32)? as *mut i32,
                cuda_ptr(&problem_sizes_s, DType::U32)? as *mut i32,
                cuda_ptr(&alphas_s, DType::F32)? as *mut f32,
                cuda_ptr(&input_scale_invs_s, DType::F32)? as *mut f32,
                num_experts as i32,
                n as i32,
                k as i32,
                stream,
            );
        }
    }

    let act_packed_ptr = pool.act_packed_ptr;
    let act_scales_ptr = pool.act_scales_ptr;
    {
        let (input_scale_invs_s, _) = input_scale_invs_t.storage_and_layout();
        let (expert_offsets_s, _) = expert_offsets_t.storage_and_layout();
        let (sf_offsets_s, _) = sf_offsets_t.storage_and_layout();
        unsafe {
            match dtype {
                DType::F16 => xinfer_nvfp4_quantize_gathered_f16(
                    cuda_ptr(&input.storage_and_layout().0, dtype)? as *const std::ffi::c_void,
                    act_packed_ptr as *mut std::ffi::c_void,
                    act_scales_ptr as *mut std::ffi::c_void,
                    cuda_ptr(&sorted_token_ids.storage_and_layout().0, DType::U32)? as *const i32,
                    map_divisor,
                    cuda_ptr(&input_scale_invs_s, DType::F32)? as *const f32,
                    cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                    cuda_ptr(&sf_offsets_s, DType::U32)? as *const i32,
                    size_m as i32,
                    num_experts as i32,
                    k as i32,
                    k_scale_padded as i32,
                    stream,
                ),
                DType::BF16 => xinfer_nvfp4_quantize_gathered_bf16(
                    cuda_ptr(&input.storage_and_layout().0, dtype)? as *const std::ffi::c_void,
                    act_packed_ptr as *mut std::ffi::c_void,
                    act_scales_ptr as *mut std::ffi::c_void,
                    cuda_ptr(&sorted_token_ids.storage_and_layout().0, DType::U32)? as *const i32,
                    map_divisor,
                    cuda_ptr(&input_scale_invs_s, DType::F32)? as *const f32,
                    cuda_ptr(&expert_offsets_s, DType::U32)? as *const i32,
                    cuda_ptr(&sf_offsets_s, DType::U32)? as *const i32,
                    size_m as i32,
                    num_experts as i32,
                    k as i32,
                    k_scale_padded as i32,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    let wss_owned;
    let weight_scales_swizzled = if let Some(pre) = pre_swizzled_weight_scales {
        pre
    } else {
        wss_owned = Tensor::zeros((num_experts, n_padded, k_scale_padded), DType::U8, dev)?;
        {
            let (ws_s, _) = weight_scales.storage_and_layout();
            let (wss_s, _) = wss_owned.storage_and_layout();
            let ws_base = cuda_ptr(&ws_s, DType::U8)?;
            let wss_base = cuda_ptr(&wss_s, DType::U8)?;
            for e in 0..num_experts {
                let src_offset = (e * n * k_scale) as u64;
                let dst_offset = (e * n_padded * k_scale_padded) as u64;
                unsafe {
                    ffi::nvfp4_swizzle_weight_scales(
                        (ws_base + src_offset) as *const std::ffi::c_void,
                        (wss_base + dst_offset) as *mut std::ffi::c_void,
                        n as i32,
                        k_scale as i32,
                        n_padded as i32,
                        k_scale_padded as i32,
                        stream,
                    );
                }
            }
        }
        &wss_owned
    };

    let rep_out_ptr = pool.rep_out_ptr;
    {
        let (weights_s, _) = weights.storage_and_layout();
        let (wss_s, _) = weight_scales_swizzled.storage_and_layout();
        let (alphas_s, _) = alphas_t.storage_and_layout();
        let (eo_s, _) = expert_offsets_t.storage_and_layout();
        let (sfo_s, _) = sf_offsets_t.storage_and_layout();
        let (ps_s, _) = problem_sizes_t.storage_and_layout();
        let (workspace_ptr, workspace_bytes) = get_moe_cutlass_workspace(cuda_dev, 0)?;
        unsafe {
            let ret = ffi::nvfp4_cutlass_moe_gemm_f32(
                rep_out_ptr as *mut std::ffi::c_void,
                act_packed_ptr as *const std::ffi::c_void,
                cuda_ptr(&weights_s, DType::U8)? as *const std::ffi::c_void,
                act_scales_ptr as *const std::ffi::c_void,
                cuda_ptr(&wss_s, DType::U8)? as *const std::ffi::c_void,
                cuda_ptr(&alphas_s, DType::F32)? as *const f32,
                cuda_ptr(&eo_s, DType::U32)? as *const i32,
                cuda_ptr(&sfo_s, DType::U32)? as *const i32,
                cuda_ptr(&ps_s, DType::U32)? as *const i32,
                num_experts as i32,
                size_m as i32,
                n as i32,
                k as i32,
                workspace_ptr,
                workspace_bytes as i64,
                stream,
            );
            if ret != 0 {
                candle_core::bail!("nvfp4_cutlass_moe_gemm failed with error code {}", ret);
            }
        }
    }

    let output = Tensor::zeros((size_m, n), dtype, dev)?;
    {
        let (sorted_s, _) = sorted_token_ids.storage_and_layout();
        let (output_s, _) = output.storage_and_layout();
        let weights_ffi_ptr = if let Some(t) = topk_weights {
            let (s, _) = t.storage_and_layout();
            cuda_ptr(&s, DType::F32)? as *const f32
        } else {
            std::ptr::null()
        };
        unsafe {
            match dtype {
                DType::F16 => ffi::moe_fp8_scatter_rows_f32_to_f16(
                    rep_out_ptr as *const f32,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    size_m as i64,
                    size_m as i64,
                    n as i64,
                    weights_ffi_ptr,
                    stream,
                ),
                DType::BF16 => ffi::moe_fp8_scatter_rows_f32_to_bf16(
                    rep_out_ptr as *const f32,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    size_m as i64,
                    size_m as i64,
                    n as i64,
                    weights_ffi_ptr,
                    stream,
                ),
                _ => unreachable!(),
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Storage};

    fn ptr(tensor: &Tensor) -> Result<u64> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            candle_core::bail!("test tensor must be contiguous and zero-offset")
        }
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("test tensor must be on CUDA")
        };
        Ok(match tensor.dtype() {
            DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
            DType::U8 => *storage.as_cuda_slice::<u8>()?.device_ptr(),
            DType::U32 => *storage.as_cuda_slice::<u32>()?.device_ptr(),
            DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
            dtype => candle_core::bail!("unsupported test dtype {dtype:?}"),
        })
    }

    #[test]
    fn fused_quant_matches_gather_then_quant_at_expert_shapes() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?) != Some(121) {
            return Ok(());
        }
        let stream = *device.as_cuda_device()?.cu_stream() as i64;
        let rows = 24usize;
        let experts = 4usize;
        let topk = 3usize;
        let sorted = Tensor::from_vec(
            (0..rows).map(|row| (row * 5 % rows) as u32).collect(),
            rows,
            &device,
        )?;
        let expert_offsets = Tensor::from_vec(vec![0u32, 3, 13, 17, 24], experts + 1, &device)?;
        let sf_offsets = Tensor::from_vec(vec![0u32, 128, 256, 384], experts, &device)?;
        let input_scale_invs = Tensor::from_vec(vec![1.0f32, 0.5, 0.25, 2.0], experts, &device)?;

        for k in [1856usize, 2688] {
            let input = Tensor::from_vec(
                (0..rows / topk * k)
                    .map(|i| half::bf16::from_f32(((i * 37 % 257) as f32 - 128.0) / 32.0))
                    .collect(),
                (rows / topk, k),
                &device,
            )?;
            let gathered = Tensor::zeros((rows, k), DType::BF16, &device)?;
            let baseline_packed = Tensor::zeros((rows, k / 2), DType::U8, &device)?;
            let fused_packed = Tensor::zeros((rows, k / 2), DType::U8, &device)?;
            let scale_cols = pad_to(k / 16, 4);
            let baseline_scales = Tensor::zeros((experts * 128 * scale_cols,), DType::U8, &device)?;
            let fused_scales = Tensor::zeros((experts * 128 * scale_cols,), DType::U8, &device)?;
            unsafe {
                ffi::nvfp4_moe_gather_bf16(
                    ptr(&input)? as *const _,
                    ptr(&gathered)? as *mut _,
                    ptr(&sorted)? as *const i32,
                    rows as i32,
                    k as i32,
                    topk as i32,
                    stream,
                );
                ffi::nvfp4_quantize_activation_grouped_bf16(
                    ptr(&gathered)? as *const _,
                    ptr(&baseline_packed)? as *mut _,
                    ptr(&baseline_scales)? as *mut _,
                    ptr(&input_scale_invs)? as *const f32,
                    ptr(&expert_offsets)? as *const i32,
                    ptr(&sf_offsets)? as *const i32,
                    rows as i32,
                    experts as i32,
                    k as i32,
                    scale_cols as i32,
                    stream,
                );
                xinfer_nvfp4_quantize_gathered_bf16(
                    ptr(&input)? as *const _,
                    ptr(&fused_packed)? as *mut _,
                    ptr(&fused_scales)? as *mut _,
                    ptr(&sorted)? as *const i32,
                    topk as i32,
                    ptr(&input_scale_invs)? as *const f32,
                    ptr(&expert_offsets)? as *const i32,
                    ptr(&sf_offsets)? as *const i32,
                    rows as i32,
                    experts as i32,
                    k as i32,
                    scale_cols as i32,
                    stream,
                );
            }
            device.synchronize()?;
            assert_eq!(
                baseline_packed.flatten_all()?.to_vec1::<u8>()?,
                fused_packed.flatten_all()?.to_vec1::<u8>()?,
                "packed E2M1 bytes differ at K={k}"
            );
            assert_eq!(
                baseline_scales.flatten_all()?.to_vec1::<u8>()?,
                fused_scales.flatten_all()?.to_vec1::<u8>()?,
                "swizzled scale bytes differ at K={k}"
            );
        }
        Ok(())
    }
}
