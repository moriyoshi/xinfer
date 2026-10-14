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

const MODULE: &str = "xinfer_nemotron_route_grouping_v2";
const COUNT: &str = "nemotron_route_count";
const PREFIX: &str = "nemotron_route_prefix";
const SCATTER: &str = "nemotron_route_scatter";
const COMBINE: &str = "nemotron_route_combine";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_ROUTE_GROUPING").as_deref() != Some(std::ffi::OsStr::new("0"))
});
static COMBINE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_FUSED_MOE_COMBINE").as_deref()
        != Some(std::ffi::OsStr::new("0"))
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
            .load_ptx(ptx, MODULE, &[COUNT, PREFIX, SCATTER, COMBINE])
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
}
