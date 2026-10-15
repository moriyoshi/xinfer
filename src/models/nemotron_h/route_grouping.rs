//! Deterministic device-side grouping of Nemotron's routed assignments.
//!
//! Generic bitonic sorting launches O(log² slots) kernels. The expert range is
//! small and fixed for a loaded model, so count, prefix and stable scatter need
//! only three launches regardless of the number of routed tokens.

use candle_core::cuda_backend::cudarc::driver::{
    capture_status, sys::CUstreamCaptureStatus, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Device, Result, Storage, Tensor};
use std::sync::LazyLock;
use std::sync::Mutex;

const MODULE: &str = "xinfer_nemotron_route_grouping_v4";
const TOPK128: &str = "nemotron_route_topk128";
const COUNT: &str = "nemotron_route_count";
const PREFIX: &str = "nemotron_route_prefix";
const SCATTER: &str = "nemotron_route_scatter";
const COMBINE: &str = "nemotron_route_combine";
const RELU2: &str = "nemotron_relu_squared";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_ROUTE_GROUPING").as_deref() != Some(std::ffi::OsStr::new("0"))
});
static COMBINE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_FUSED_MOE_COMBINE").as_deref()
        != Some(std::ffi::OsStr::new("0"))
});
static RELU2_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_FUSED_RELU2").as_deref() != Some(std::ffi::OsStr::new("0"))
});
static TOPK128_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_TOPK_ONE_KERNEL").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    slots: i32,
    experts: i32,
}

unsafe impl DeviceRepr for Params {}

#[repr(C)]
#[derive(Clone, Copy)]
struct CombineParams {
    rows: i32,
    topk: i32,
    hidden: i32,
    dtype: i32,
}

unsafe impl DeviceRepr for CombineParams {}

fn ensure_module(device: &Device) -> Result<()> {
    let cuda = device.as_cuda_device()?;
    let driver = cuda.cuda_device();
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !driver.has_func(MODULE, COUNT) {
        let capture = capture_status(*driver.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Nemotron CUDA capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Nemotron routing CUDA kernels must be loaded before graph capture")
        }
        let sm = attention_rs::cuda_utils::sm_version(cuda)
            .ok_or_else(|| candle_core::Error::Msg("unknown CUDA compute capability".into()))?;
        let ptx = compile_ptx_with_opts(
            include_str!("route_grouping.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile Nemotron routes: {e}")))?;
        driver
            .load_ptx(
                ptx,
                MODULE,
                &[COUNT, PREFIX, SCATTER, COMBINE, RELU2, TOPK128],
            )
            .map_err(|e| candle_core::Error::Msg(format!("load Nemotron routes: {e}")))?;
    }
    Ok(())
}

fn pointer(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    if tensor.dtype() != DType::U32 || !layout.is_contiguous() {
        candle_core::bail!("Nemotron route grouping requires contiguous U32 tensors")
    }
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Nemotron route grouping requires CUDA tensors")
    };
    let offset = layout
        .start_offset()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| candle_core::Error::Msg("Nemotron route offset overflow".into()))?;
    (*storage.as_cuda_slice::<u32>()?.device_ptr())
        .checked_add(offset as u64)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron route pointer overflow".into()))
}

fn float_pointer(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    if !layout.is_contiguous() {
        candle_core::bail!("Nemotron MoE combine requires contiguous tensors")
    }
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Nemotron MoE combine requires CUDA tensors")
    };
    let base = match tensor.dtype() {
        DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
        DType::F16 => *storage.as_cuda_slice::<half::f16>()?.device_ptr(),
        DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
        dtype => candle_core::bail!("unsupported Nemotron MoE combine dtype {dtype:?}"),
    };
    let offset = layout
        .start_offset()
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| candle_core::Error::Msg("Nemotron MoE pointer offset overflow".into()))?;
    base.checked_add(offset as u64)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron MoE pointer overflow".into()))
}

/// Select the same expert IDs as the generic descending bitonic sort for the
/// fixed 128-expert Nemotron layout. The sigmoid, correction bias, gather and
/// normalization stay in the shared routing implementation.
pub(super) fn topk128(scores: &Tensor, topk: usize) -> Result<Option<Tensor>> {
    if !*TOPK128_ENABLED {
        return Ok(None);
    }
    topk128_kernel(scores, topk)
}

fn topk128_kernel(scores: &Tensor, topk: usize) -> Result<Option<Tensor>> {
    if !scores.device().is_cuda() || scores.dtype() != DType::F32 {
        return Ok(None);
    }
    let (rows, experts) = scores.dims2()?;
    if rows == 0 || rows > i32::MAX as usize || experts != 128 || !(1..=128).contains(&topk) {
        return Ok(None);
    }
    let scores = scores.contiguous()?;
    let driver = scores.device().as_cuda_device()?.cuda_device();
    if !driver.has_func(MODULE, TOPK128)
        && capture_status(*driver.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Nemotron CUDA capture: {e}")))?
            == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    {
        return Ok(None);
    }
    ensure_module(scores.device())?;
    // Every output element is written by one lane before its first consumer.
    let ids = unsafe { Tensor::empty_((rows, topk), DType::U32, scores.device())? };
    let kernel = driver
        .get_func(MODULE, TOPK128)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron top-k kernel".into()))?;
    unsafe {
        kernel.launch(
            LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                float_pointer(&scores)?,
                pointer(&ids)?,
                rows as i32,
                topk as i32,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron top-k: {e}")))?;
    Ok(Some(ids))
}

/// Returns expert IDs and original flattened assignment indices, grouped by
/// expert. Within each expert, indices remain in their original order.
pub(super) fn sort_expert_assignments(
    topk_ids: &Tensor,
    num_experts: usize,
) -> Result<(Tensor, Tensor)> {
    let flat = topk_ids.flatten_all()?.contiguous()?;
    let slots = flat.elem_count();
    if slots == 0 || num_experts == 0 {
        candle_core::bail!("Nemotron route grouping needs nonempty assignments and experts")
    }
    if !*ENABLED || num_experts > 256 || slots > i32::MAX as usize || flat.dtype() != DType::U32 {
        return crate::models::layers::moe::sort_expert_assignments(topk_ids, true);
    }
    ensure_module(flat.device())?;
    let device = flat.device().as_cuda_device()?.cuda_device();
    let counts = Tensor::zeros((num_experts,), DType::U32, flat.device())?;
    let offsets = Tensor::zeros((num_experts + 1,), DType::U32, flat.device())?;
    let expert_ids = Tensor::zeros((slots,), DType::U32, flat.device())?;
    let assignment_ids = Tensor::zeros((slots,), DType::U32, flat.device())?;
    let params = Params {
        slots: slots as i32,
        experts: num_experts as i32,
    };
    let count = device
        .get_func(MODULE, COUNT)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron route count kernel".into()))?;
    unsafe {
        count.launch(
            LaunchConfig {
                grid_dim: (num_experts as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (pointer(&flat)?, pointer(&counts)?, params),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron route count: {e}")))?;
    let prefix = device
        .get_func(MODULE, PREFIX)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron route prefix kernel".into()))?;
    unsafe {
        prefix.launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (pointer(&counts)?, pointer(&offsets)?, params),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron route prefix: {e}")))?;
    let scatter = device
        .get_func(MODULE, SCATTER)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron route scatter kernel".into()))?;
    unsafe {
        scatter.launch(
            LaunchConfig {
                grid_dim: (num_experts as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer(&flat)?,
                pointer(&offsets)?,
                pointer(&expert_ids)?,
                pointer(&assignment_ids)?,
                params,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron route scatter: {e}")))?;
    Ok((expert_ids, assignment_ids))
}

/// Return `None` when the original Candle combination is required.
pub(super) fn combine_routed(
    routed: &Tensor,
    shared: &Tensor,
    weights: &Tensor,
) -> Result<Option<Tensor>> {
    let (rows, topk, hidden) = routed.dims3()?;
    if shared.dims() != [rows, hidden] || weights.dims() != [rows, topk] {
        candle_core::bail!("Nemotron MoE combine shape mismatch")
    }
    if !*COMBINE_ENABLED
        || !matches!(routed.dtype(), DType::BF16 | DType::F16)
        || shared.dtype() != routed.dtype()
        || weights.dtype() != DType::F32
        || !routed.is_contiguous()
        || !shared.is_contiguous()
        || !weights.is_contiguous()
        || topk > 8
        || rows.saturating_mul(hidden) > i32::MAX as usize
    {
        return Ok(None);
    }
    let cuda = routed.device().as_cuda_device()?.cuda_device();
    if !cuda.has_func(MODULE, COMBINE)
        && capture_status(*cuda.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Nemotron CUDA capture: {e}")))?
            == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    {
        return Ok(None);
    }
    ensure_module(routed.device())?;
    let device = cuda;
    let output = Tensor::zeros((rows, hidden), routed.dtype(), routed.device())?;
    let params = CombineParams {
        rows: rows as i32,
        topk: topk as i32,
        hidden: hidden as i32,
        dtype: if routed.dtype() == DType::BF16 { 0 } else { 1 },
    };
    let kernel = device
        .get_func(MODULE, COMBINE)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron MoE combine kernel".into()))?;
    unsafe {
        kernel.launch(
            LaunchConfig {
                grid_dim: ((rows * hidden).div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                float_pointer(routed)?,
                float_pointer(shared)?,
                float_pointer(weights)?,
                float_pointer(&output)?,
                params,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron MoE combine: {e}")))?;
    Ok(Some(output))
}

/// Fuse the two elementwise passes of ReLU-squared on CUDA. Returns `None`
/// when the original Candle operations should be used.
pub(super) fn relu_squared(input: &Tensor) -> Result<Option<Tensor>> {
    let elements = input.elem_count();
    if !*RELU2_ENABLED
        || elements == 0
        || elements > i32::MAX as usize
        || !input.is_contiguous()
        || !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32)
        || !input.device().is_cuda()
    {
        return Ok(None);
    }
    let driver = input.device().as_cuda_device()?.cuda_device();
    if !driver.has_func(MODULE, RELU2)
        && capture_status(*driver.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Nemotron CUDA capture: {e}")))?
            == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    {
        return Ok(None);
    }
    ensure_module(input.device())?;
    // The kernel writes each output element exactly once before any consumer.
    let output = unsafe { Tensor::empty_(input.shape(), input.dtype(), input.device())? };
    let kernel = driver
        .get_func(MODULE, RELU2)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron ReLU-squared kernel".into()))?;
    let dtype = match input.dtype() {
        DType::BF16 => 0,
        DType::F16 => 1,
        DType::F32 => 2,
        _ => unreachable!(),
    };
    unsafe {
        kernel.launch(
            LaunchConfig {
                grid_dim: (elements.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                float_pointer(input)?,
                float_pointer(&output)?,
                elements as i32,
                dtype,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron ReLU-squared: {e}")))?;
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use attention_rs::sort::ArgSortOp;
    use candle_core::Device;

    #[test]
    fn one_kernel_topk_matches_generic_bitonic_sort() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for rows in [1, 17, 257, 8192] {
            for case in 0..3 {
                let values = (0..rows * 128)
                    .map(|i| match case {
                        0 => ((i * 73 + i / 7 * 19) % 1009) as f32 / 1009.0,
                        1 => ((i / 9) % 7) as f32,
                        _ if i % 113 == 0 => f32::NAN,
                        _ => ((i * 17) % 13) as f32,
                    })
                    .collect::<Vec<_>>();
                let scores = Tensor::from_vec(values, (rows, 128), &device)?;
                let expected = scores.arg_sort(false)?.narrow(1, 0, 6)?.to_vec2::<u32>()?;
                let actual = topk128_kernel(&scores, 6)?.unwrap().to_vec2::<u32>()?;
                assert_eq!(actual, expected, "rows={rows} case={case}");
            }
        }
        Ok(())
    }

    #[test]
    fn stable_route_grouping_matches_cpu_oracle() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for (rows, experts, topk) in [(1, 128, 6), (17, 128, 6), (513, 128, 6), (8192, 128, 6)] {
            let ids: Vec<u32> = (0..rows * topk)
                .map(|i| ((i * 73 + i / 7 * 19) % experts) as u32)
                .collect();
            let input = Tensor::from_vec(ids.clone(), (rows, topk), &device)?;
            let (sorted_experts, sorted_assignments) = sort_expert_assignments(&input, experts)?;
            let sorted_experts = sorted_experts.to_vec1::<u32>()?;
            let sorted_assignments = sorted_assignments.to_vec1::<u32>()?;
            let mut oracle: Vec<usize> = (0..ids.len()).collect();
            oracle.sort_by_key(|&i| ids[i]);
            assert_eq!(
                sorted_assignments,
                oracle.iter().map(|&i| i as u32).collect::<Vec<_>>()
            );
            assert_eq!(
                sorted_experts,
                oracle.iter().map(|&i| ids[i]).collect::<Vec<_>>()
            );
        }
        for expert in [0u32, 127] {
            let input = Tensor::from_vec(vec![expert; 513 * 6], (513, 6), &device)?;
            let (sorted_experts, sorted_assignments) = sort_expert_assignments(&input, 128)?;
            assert_eq!(sorted_experts.to_vec1::<u32>()?, vec![expert; 513 * 6]);
            assert_eq!(
                sorted_assignments.to_vec1::<u32>()?,
                (0..513 * 6).map(|i| i as u32).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn fused_combine_matches_candle_rounding() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (rows, topk, hidden) = (17, 6, 257);
        let routed = Tensor::from_vec(
            (0..rows * topk * hidden)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 29.0)
                .collect::<Vec<_>>(),
            (rows, topk, hidden),
            &device,
        )?;
        let shared = Tensor::from_vec(
            (0..rows * hidden)
                .map(|i| ((i * 19 % 73) as f32 - 36.0) / 17.0)
                .collect::<Vec<_>>(),
            (rows, hidden),
            &device,
        )?;
        let weights = Tensor::from_vec(
            (0..rows * topk)
                .map(|i| (i % topk + 1) as f32 / 32.0)
                .collect::<Vec<_>>(),
            (rows, topk),
            &device,
        )?;
        for dtype in [DType::BF16, DType::F16] {
            let routed = routed.to_dtype(dtype)?;
            let shared = shared.to_dtype(dtype)?;
            let expected = routed
                .to_dtype(DType::F32)?
                .broadcast_mul(&weights.unsqueeze(2)?)?
                .sum(1)?
                .broadcast_add(&shared.to_dtype(DType::F32)?)?
                .to_dtype(dtype)?;
            let actual = combine_routed(&routed, &shared, &weights)?.unwrap();
            match dtype {
                DType::BF16 => assert_eq!(
                    expected.flatten_all()?.to_vec1::<half::bf16>()?,
                    actual.flatten_all()?.to_vec1::<half::bf16>()?
                ),
                DType::F16 => assert_eq!(
                    expected.flatten_all()?.to_vec1::<half::f16>()?,
                    actual.flatten_all()?.to_vec1::<half::f16>()?
                ),
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    #[test]
    fn fused_relu_squared_matches_candle_for_expert_shapes() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for shape in [(17, 1, 257), (9, 6, 97)] {
            let elements = shape.0 * shape.1 * shape.2;
            let values = (0..elements)
                .map(|i| ((i * 37 % 113) as f32 - 56.0) / 29.0)
                .collect::<Vec<_>>();
            let base = Tensor::from_vec(values, shape, &device)?;
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let input = base.to_dtype(dtype)?;
                let expected = input.relu()?.sqr()?.flatten_all()?;
                let actual = relu_squared(&input)?.unwrap().flatten_all()?;
                match dtype {
                    DType::BF16 => assert_eq!(
                        expected.to_vec1::<half::bf16>()?,
                        actual.to_vec1::<half::bf16>()?
                    ),
                    DType::F16 => assert_eq!(
                        expected.to_vec1::<half::f16>()?,
                        actual.to_vec1::<half::f16>()?
                    ),
                    DType::F32 => assert_eq!(expected.to_vec1::<f32>()?, actual.to_vec1::<f32>()?),
                    _ => unreachable!(),
                }
            }
        }
        let noncontiguous = Tensor::zeros((3, 5), DType::BF16, &device)?.transpose(0, 1)?;
        assert!(relu_squared(&noncontiguous)?.is_none());
        Ok(())
    }
}
