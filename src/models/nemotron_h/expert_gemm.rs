//! Xinfer-owned fused expert gather/activation quantization for Nemotron-H.
//! Based on the pinned attention-rs grouped NVFP4 hardware wrapper (MIT).
//! Activation quantization reads routed rows directly from their source
//! instead of materializing a BF16 gather buffer. An opt-in kernel variant
//! also applies ReLU² with the original BF16 rounding during that read.

use attention_rs::workspace::{get_moe_activation_pool, get_moe_cutlass_workspace};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Result, Tensor};
use kernels::ffi;
use std::sync::LazyLock;

static VECTOR_SCATTER: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_EXPERT_VECTOR_SCATTER").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});

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
        relu_squared: i32,
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
        relu_squared: i32,
        stream: i64,
    );
    fn xinfer_nvfp4_scatter_f32_to_bf16_vec(
        input: *const f32,
        sorted_ids: *const i32,
        output: *mut std::ffi::c_void,
        weights: *const f32,
        rows: i32,
        columns: i32,
        stream: i64,
    ) -> i32;
    fn xinfer_nvfp4_finalize_bf16(
        routed: *const f32,
        sorted_ids: *const i32,
        inverse: *mut i32,
        shared: *const std::ffi::c_void,
        weights: *const f32,
        output: *mut std::ffi::c_void,
        rows: i32,
        topk: i32,
        hidden: i32,
        stream: i64,
    ) -> i32;
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
    finalize: Option<(&Tensor, &Tensor)>,
    relu_squared: bool,
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

    // calculate_expert_offsets initializes counts itself and writes every
    // offset. nvfp4_moe_build_metadata then writes every word of the remaining
    // buffers on this stream. Zero-filling here adds six device memsets per
    // expert GEMM without providing any initialized data to a consumer.
    let expert_counts_t = unsafe { Tensor::empty_((num_experts,), DType::U32, dev)? };
    let expert_offsets_t = unsafe { Tensor::empty_((num_experts + 1,), DType::U32, dev)? };
    let sf_offsets_t = unsafe { Tensor::empty_((num_experts,), DType::U32, dev)? };
    let problem_sizes_t = unsafe { Tensor::empty_((num_experts * 3,), DType::U32, dev)? };
    let alphas_t = unsafe { Tensor::empty_((num_experts,), DType::F32, dev)? };
    let input_scale_invs_t = unsafe { Tensor::empty_((num_experts,), DType::F32, dev)? };

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
                    i32::from(relu_squared),
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
                    i32::from(relu_squared),
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

    if let Some((shared, route_weights)) = finalize {
        if topk == 0 || size_m % topk != 0 {
            candle_core::bail!("Nemotron fused expert finalizer requires complete routes")
        }
        let rows = size_m / topk;
        if topk > 8
            || size_m > i32::MAX as usize
            || n > i32::MAX as usize
            || rows
                .checked_mul(n)
                .is_none_or(|count| count > i32::MAX as usize)
            || shared.dims() != [rows, n]
            || shared.dtype() != DType::BF16
            || !shared.is_contiguous()
            || shared.storage_and_layout().1.start_offset() != 0
            || route_weights.dims() != [rows, topk]
            || route_weights.dtype() != DType::F32
            || !route_weights.is_contiguous()
            || route_weights.storage_and_layout().1.start_offset() != 0
            || dtype != DType::BF16
        {
            candle_core::bail!("Nemotron fused expert finalizer shape or dtype mismatch")
        }
        // The private route sorter produces a permutation of every assignment.
        // Inversion and finalization run on the same stream, so no zero fill is
        // needed for either buffer.
        let inverse = unsafe { Tensor::empty_((size_m,), DType::U32, dev)? };
        let output = unsafe { Tensor::empty_((rows, n), DType::BF16, dev)? };
        let code = unsafe {
            xinfer_nvfp4_finalize_bf16(
                rep_out_ptr as *const f32,
                cuda_ptr(&sorted_token_ids.storage_and_layout().0, DType::U32)? as *const i32,
                cuda_ptr(&inverse.storage_and_layout().0, DType::U32)? as *mut i32,
                cuda_ptr(&shared.storage_and_layout().0, DType::BF16)? as *const _,
                cuda_ptr(&route_weights.storage_and_layout().0, DType::F32)? as *const f32,
                cuda_ptr(&output.storage_and_layout().0, DType::BF16)? as *mut _,
                rows as i32,
                topk as i32,
                n as i32,
                stream,
            )
        };
        if code != 0 {
            candle_core::bail!("Nemotron fused expert finalizer launch failed: {code}")
        }
        return Ok(output);
    }

    // The grouped route IDs are a permutation of all assignment rows. Every
    // output word is written once, so the vector path can skip a full zero
    // fill. Keep the old initialized allocation on the fallback path.
    let vector_scatter = *VECTOR_SCATTER
        && dtype == DType::BF16
        && n.is_multiple_of(4)
        && n <= i32::MAX as usize
        && size_m <= i32::MAX as usize;
    let output = if vector_scatter {
        unsafe { Tensor::empty_((size_m, n), dtype, dev)? }
    } else {
        Tensor::zeros((size_m, n), dtype, dev)?
    };
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
            if vector_scatter {
                let code = xinfer_nvfp4_scatter_f32_to_bf16_vec(
                    rep_out_ptr as *const f32,
                    cuda_ptr(&sorted_s, DType::U32)? as *const i32,
                    cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void,
                    weights_ffi_ptr,
                    size_m as i32,
                    n as i32,
                    stream,
                );
                if code != 0 {
                    candle_core::bail!("Nemotron vector expert scatter launch failed: {code}")
                }
            } else {
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
                    0,
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
            let activated = super::super::relu_squared(&input)?;
            unsafe {
                ffi::nvfp4_moe_gather_bf16(
                    ptr(&activated)? as *const _,
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
                    1,
                    stream,
                );
            }
            device.synchronize()?;
            assert_eq!(
                baseline_packed.flatten_all()?.to_vec1::<u8>()?,
                fused_packed.flatten_all()?.to_vec1::<u8>()?,
                "ReLU-squared packed bytes differ at K={k}"
            );
            assert_eq!(
                baseline_scales.flatten_all()?.to_vec1::<u8>()?,
                fused_scales.flatten_all()?.to_vec1::<u8>()?,
                "ReLU-squared scale bytes differ at K={k}"
            );
        }
        Ok(())
    }

    #[test]
    fn vector_scatter_matches_reference_bytes_for_permuted_routes() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?) != Some(121) {
            return Ok(());
        }
        let stream = *device.as_cuda_device()?.cu_stream() as i64;
        let rows = 37usize;
        let sorted = Tensor::from_vec(
            (0..rows).map(|i| ((i * 13) % rows) as u32).collect(),
            rows,
            &device,
        )?;
        let weights = Tensor::from_vec(
            (0..rows)
                .map(|i| ((i * 19 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            rows,
            &device,
        )?;
        for columns in [1856usize, 2688] {
            let input = Tensor::from_vec(
                (0..rows * columns)
                    .map(|i| ((i * 107 % 401) as f32 - 200.0) / 37.0)
                    .collect(),
                (rows, columns),
                &device,
            )?;
            let reference = Tensor::zeros((rows, columns), DType::BF16, &device)?;
            let actual = Tensor::zeros((rows, columns), DType::BF16, &device)?;
            unsafe {
                ffi::moe_fp8_scatter_rows_f32_to_bf16(
                    ptr(&input)? as *const f32,
                    ptr(&sorted)? as *const i32,
                    ptr(&reference)? as *mut _,
                    rows as i64,
                    rows as i64,
                    columns as i64,
                    ptr(&weights)? as *const f32,
                    stream,
                );
                let code = xinfer_nvfp4_scatter_f32_to_bf16_vec(
                    ptr(&input)? as *const f32,
                    ptr(&sorted)? as *const i32,
                    ptr(&actual)? as *mut _,
                    ptr(&weights)? as *const f32,
                    rows as i32,
                    columns as i32,
                    stream,
                );
                assert_eq!(code, 0);
            }
            device.synchronize()?;
            let bits = |tensor: &Tensor| -> Result<Vec<u16>> {
                Ok(tensor
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .iter()
                    .map(|value| value.to_bits())
                    .collect())
            };
            assert_eq!(bits(&reference)?, bits(&actual)?, "width {columns}");
        }
        Ok(())
    }

    #[test]
    fn fused_finalizer_matches_scatter_and_combine_bytes() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?) != Some(121) {
            return Ok(());
        }
        let rows = 19usize;
        let topk = 6usize;
        let hidden = 2688usize;
        let assignments = rows * topk;
        let stream = *device.as_cuda_device()?.cu_stream() as i64;
        let source = Tensor::from_vec(
            (0..assignments * hidden)
                .map(|i| ((i * 107 % 401) as f32 - 200.0) / 37.0)
                .collect(),
            (assignments, hidden),
            &device,
        )?;
        let sorted = Tensor::from_vec(
            (0..assignments)
                .map(|i| ((i * 13) % assignments) as u32)
                .collect(),
            assignments,
            &device,
        )?;
        let shared = Tensor::from_vec(
            (0..rows * hidden)
                .map(|i| ((i * 31 % 251) as f32 - 125.0) / 43.0)
                .collect(),
            (rows, hidden),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weights = Tensor::from_vec(
            (0..assignments)
                .map(|i| ((i * 19 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            (rows, topk),
            &device,
        )?;
        let scattered = Tensor::zeros((assignments, hidden), DType::BF16, &device)?;
        unsafe {
            ffi::moe_fp8_scatter_rows_f32_to_bf16(
                ptr(&source)? as *const f32,
                ptr(&sorted)? as *const i32,
                ptr(&scattered)? as *mut _,
                assignments as i64,
                assignments as i64,
                hidden as i64,
                std::ptr::null(),
                stream,
            );
        }
        let expected = super::super::route_grouping::combine_routed(
            &scattered.reshape((rows, topk, hidden))?,
            &shared,
            &weights,
        )?
        .ok_or_else(|| candle_core::Error::Msg("fused route combine unavailable".into()))?;
        let inverse = unsafe { Tensor::empty_((assignments,), DType::U32, &device)? };
        let actual = unsafe { Tensor::empty_((rows, hidden), DType::BF16, &device)? };
        let code = unsafe {
            xinfer_nvfp4_finalize_bf16(
                ptr(&source)? as *const f32,
                ptr(&sorted)? as *const i32,
                ptr(&inverse)? as *mut i32,
                ptr(&shared)? as *const _,
                ptr(&weights)? as *const f32,
                ptr(&actual)? as *mut _,
                rows as i32,
                topk as i32,
                hidden as i32,
                stream,
            )
        };
        assert_eq!(code, 0);
        device.synchronize()?;
        let bits = |tensor: &Tensor| -> Result<Vec<u16>> {
            Ok(tensor
                .flatten_all()?
                .to_vec1::<half::bf16>()?
                .iter()
                .map(|value| value.to_bits())
                .collect())
        };
        assert_eq!(bits(&expected)?, bits(&actual)?);
        Ok(())
    }
}
