//! CUDA span prefill for the Nemotron-H Mamba-2 mixer. A preparation kernel
//! computes causal convolution and dt for every row. One block per head then
//! keeps the FP32 SSM state in registers while scanning the span. Gating,
//! grouped normalization and output projection run as batched Candle work.

use super::{Mamba2, MambaState};
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::Mutex;

const PREPARE: &str = "nemotron_mamba_prepare";
const SCAN: &str = "nemotron_mamba_scan";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    tokens: i32,
    heads: i32,
    head_dim: i32,
    groups: i32,
    state_dim: i32,
    kernel: i32,
    conv_dim: i32,
    in_dim: i32,
    projected_dtype: i32, // 0 BF16, 1 F16, 2 F32
    dt_min: f32,
    dt_max: f32,
}

unsafe impl DeviceRepr for Params {}

fn as_i32(value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("Nemotron Mamba prefill dimension exceeds i32".into()))
}

fn as_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("Nemotron Mamba prefill grid exceeds u32".into()))
}

fn pointer(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    if !layout.is_contiguous() {
        candle_core::bail!("Nemotron Mamba prefill tensor must be contiguous")
    }
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Nemotron Mamba prefill tensor must be CUDA")
    };
    let (address, word_bytes) = match tensor.dtype() {
        DType::BF16 => (*storage.as_cuda_slice::<half::bf16>()?.device_ptr(), 2usize),
        DType::F16 => (*storage.as_cuda_slice::<half::f16>()?.device_ptr(), 2),
        DType::F32 => (*storage.as_cuda_slice::<f32>()?.device_ptr(), 4),
        dtype => candle_core::bail!("unsupported Nemotron Mamba prefill dtype {dtype:?}"),
    };
    let offset = layout
        .start_offset()
        .checked_mul(word_bytes)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron Mamba pointer offset overflow".into()))?;
    address
        .checked_add(offset as u64)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron Mamba device address overflow".into()))
}

pub(super) fn supported(mixer: &Mamba2, projected: &Tensor) -> bool {
    if !mixer.device.is_cuda() {
        return false;
    }
    let Ok(cuda) = mixer.device.as_cuda_device() else {
        return false;
    };
    if !matches!(capture_status(*cuda.cuda_device().cu_stream()), Ok(status) if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE)
    {
        return false;
    }
    mixer.state_dim == 128
        && mixer.head_dim.is_multiple_of(8)
        && mixer.head_dim <= 128
        && matches!(projected.dtype(), DType::BF16 | DType::F16 | DType::F32)
}

pub(super) fn run(mixer: &Mamba2, projected: &Tensor, state: &mut MambaState) -> Result<Tensor> {
    let tokens = projected.dim(0)?;
    let inner = mixer.heads * mixer.head_dim;
    let in_dim = inner + mixer.conv_dim + mixer.heads;
    if tokens < 2
        || projected.dim(1)? != in_dim
        || state.conv.dims() != [mixer.kernel - 1, mixer.conv_dim]
        || state.ssm.dims() != [mixer.heads, mixer.head_dim, mixer.state_dim]
        || !supported(mixer, projected)
    {
        candle_core::bail!("unsupported Nemotron Mamba prefill shape")
    }
    let device = mixer.device.as_cuda_device()?.cuda_device();
    let capture = capture_status(*device.cu_stream())
        .map_err(|e| candle_core::Error::Msg(format!("check Mamba CUDA capture: {e}")))?;
    if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        candle_core::bail!("Nemotron Mamba span prefill cannot run during CUDA graph capture")
    }
    let module = format!("xinfer_nemotron_mamba_prefill_h{}_v1", mixer.head_dim);
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !device.has_func(&module, PREPARE) {
            let source = format!(
                "#define HEAD_DIM {}\n{}",
                mixer.head_dim,
                include_str!("prefill/scan.cu")
            );
            let ptx = compile_ptx(source)
                .map_err(|e| candle_core::Error::Msg(format!("compile Mamba prefill: {e}")))?;
            device
                .load_ptx(ptx, &module, &[PREPARE, SCAN])
                .map_err(|e| candle_core::Error::Msg(format!("load Mamba prefill PTX: {e}")))?;
        }
    }
    let projected = projected.contiguous()?;
    let conv = Tensor::zeros((tokens, mixer.conv_dim), DType::F32, &mixer.device)?;
    let next_conv = Tensor::zeros(
        (mixer.kernel - 1, mixer.conv_dim),
        DType::F32,
        &mixer.device,
    )?;
    let dt_decay = Tensor::zeros((tokens, mixer.heads, 2), DType::F32, &mixer.device)?;
    let y = Tensor::zeros((tokens, inner), DType::F32, &mixer.device)?;
    let next_ssm = Tensor::zeros(
        (mixer.heads, mixer.head_dim, mixer.state_dim),
        DType::F32,
        &mixer.device,
    )?;
    let params = Params {
        tokens: as_i32(tokens)?,
        heads: as_i32(mixer.heads)?,
        head_dim: as_i32(mixer.head_dim)?,
        groups: as_i32(mixer.groups)?,
        state_dim: as_i32(mixer.state_dim)?,
        kernel: as_i32(mixer.kernel)?,
        conv_dim: as_i32(mixer.conv_dim)?,
        in_dim: as_i32(in_dim)?,
        projected_dtype: match projected.dtype() {
            DType::BF16 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
            _ => unreachable!(),
        },
        dt_min: mixer.dt_limit.0 as f32,
        dt_max: mixer.dt_limit.1 as f32,
    };
    let work = tokens
        .checked_mul(mixer.conv_dim)
        .and_then(|v| {
            (mixer.kernel - 1)
                .checked_mul(mixer.conv_dim)
                .and_then(|history| v.checked_add(history))
        })
        .and_then(|v| {
            tokens
                .checked_mul(mixer.heads)
                .and_then(|dt| v.checked_add(dt))
        })
        .ok_or_else(|| candle_core::Error::Msg("Nemotron Mamba prefill work overflow".into()))?;
    let prepare = device
        .get_func(&module, PREPARE)
        .ok_or_else(|| candle_core::Error::Msg("missing Mamba prepare kernel".into()))?;
    unsafe {
        prepare.launch(
            LaunchConfig {
                grid_dim: (as_u32(work.div_ceil(256))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer(&projected)?,
                pointer(&state.conv)?,
                pointer(&mixer.conv_weight)?,
                pointer(&mixer.conv_bias)?,
                pointer(&mixer.dt_bias)?,
                pointer(&mixer.a)?,
                pointer(&conv)?,
                pointer(&next_conv)?,
                pointer(&dt_decay)?,
                params,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Mamba prepare: {e}")))?;
    let scan = device
        .get_func(&module, SCAN)
        .ok_or_else(|| candle_core::Error::Msg("missing Mamba scan kernel".into()))?;
    unsafe {
        scan.launch(
            LaunchConfig {
                grid_dim: (as_u32(mixer.heads)?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer(&conv)?,
                pointer(&dt_decay)?,
                pointer(&state.ssm)?,
                pointer(&mixer.d)?,
                pointer(&y)?,
                pointer(&next_ssm)?,
                params,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Mamba scan: {e}")))?;
    device
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("synchronize Mamba scan: {e}")))?;
    state.conv = next_conv;
    state.ssm = next_ssm;

    let gate = projected.narrow(1, 0, inner)?.to_dtype(DType::F32)?;
    let gated = y.broadcast_mul(&candle_nn::ops::silu(&gate)?)?;
    let grouped = gated.reshape((tokens, mixer.groups, inner / mixer.groups))?;
    let variance = grouped.sqr()?.mean_keepdim(2)?;
    let normalized = grouped
        .broadcast_div(&(variance + mixer.eps)?.sqrt()?)?
        .reshape((tokens, inner))?
        .broadcast_mul(&mixer.norm_weight)?;
    mixer.out_proj.forward(&normalized.to_dtype(mixer.dtype)?)
}

#[cfg(test)]
mod tests;
