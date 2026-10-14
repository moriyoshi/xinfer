//! CUDA span prefill for the Nemotron-H Mamba-2 mixer. A preparation kernel
//! computes causal convolution and dt for every row. One block per head then
//! keeps the FP32 SSM state in registers while scanning the span. Gating,
//! grouped normalization and output projection run as batched Candle work.

use super::{Mamba2, MambaState};
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::LazyLock;
use std::sync::Mutex;

const PREPARE: &str = "nemotron_mamba_prepare";
const SCAN: &str = "nemotron_mamba_scan";
const CHUNK_SUMMARY: &str = "nemotron_mamba_chunk_summary";
const CHUNK_PREFIX: &str = "nemotron_mamba_chunk_prefix";
const CHUNK_SCAN: &str = "nemotron_mamba_chunk_scan";
const GATE_NORM: &str = "nemotron_mamba_gate_norm";
const STEP: &str = "nemotron_mamba_step";
const CHUNK_TOKENS: usize = 256;
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static CHUNKED_SCAN: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_CHUNKED_SCAN").as_deref() != Some(std::ffi::OsStr::new("0"))
});
static FUSED_GATE_NORM: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_FUSED_GATE_NORM").as_deref()
        != Some(std::ffi::OsStr::new("0"))
});
static SYNC_SCAN: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_PREFILL_SCAN_SYNC").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});

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
    output_dtype: i32,
    dt_min: f32,
    dt_max: f32,
    eps: f32,
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

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::BF16 => Ok(0),
        DType::F16 => Ok(1),
        DType::F32 => Ok(2),
        _ => candle_core::bail!("unsupported Nemotron Mamba CUDA dtype {dtype:?}"),
    }
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

pub(super) fn supported_step(mixer: &Mamba2, projected: &Tensor) -> bool {
    let sm = mixer
        .device
        .as_cuda_device()
        .ok()
        .and_then(|device| attention_rs::cuda_utils::sm_version(&device.cuda_device()))
        .unwrap_or(0);
    mixer.device.is_cuda()
        && sm >= 80
        && mixer.state_dim == 128
        && mixer.kernel >= 2
        && mixer.groups > 0
        && mixer.heads.is_multiple_of(mixer.groups)
        && mixer.heads * mixer.head_dim / mixer.groups >= 256
        && matches!(projected.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && matches!(mixer.dtype, DType::BF16 | DType::F16 | DType::F32)
}

fn module(mixer: &Mamba2) -> Result<String> {
    let device = mixer.device.as_cuda_device()?.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(mixer.device.as_cuda_device()?)
        .ok_or_else(|| candle_core::Error::Msg("unknown Mamba CUDA compute capability".into()))?;
    let module = format!("xinfer_nemotron_mamba_h{}_sm{}_v6", mixer.head_dim, sm);
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !device.has_func(&module, PREPARE) {
        let capture = capture_status(*device.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Mamba CUDA capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Nemotron Mamba CUDA kernels must be loaded before graph capture")
        }
        let source = format!(
            "#define HEAD_DIM {}\n{}",
            mixer.head_dim,
            include_str!("prefill/scan.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile Mamba CUDA: {e}")))?;
        device
            .load_ptx(
                ptx,
                &module,
                &[
                    PREPARE,
                    SCAN,
                    CHUNK_SUMMARY,
                    CHUNK_PREFIX,
                    CHUNK_SCAN,
                    GATE_NORM,
                    STEP,
                ],
            )
            .map_err(|e| candle_core::Error::Msg(format!("load Mamba CUDA PTX: {e}")))?;
    }
    Ok(module)
}

pub(super) fn run(mixer: &Mamba2, projected: &Tensor, state: &mut MambaState) -> Result<Tensor> {
    run_with_options(mixer, projected, state, *CHUNKED_SCAN, *FUSED_GATE_NORM)
}

fn run_with_options(
    mixer: &Mamba2,
    projected: &Tensor,
    state: &mut MambaState,
    chunked_scan: bool,
    fused_gate_norm: bool,
) -> Result<Tensor> {
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
    let module = module(mixer)?;
    let projected = projected.contiguous()?;
    // Every element of these buffers is written by prepare/scan before the
    // next kernel reads it. All launches use the same CUDA stream.
    let conv = unsafe { Tensor::empty_((tokens, mixer.conv_dim), DType::F32, &mixer.device)? };
    let next_conv = unsafe {
        Tensor::empty_(
            (mixer.kernel - 1, mixer.conv_dim),
            DType::F32,
            &mixer.device,
        )?
    };
    let dt_decay = unsafe { Tensor::empty_((tokens, mixer.heads, 2), DType::F32, &mixer.device)? };
    let y = unsafe { Tensor::empty_((tokens, inner), DType::F32, &mixer.device)? };
    let next_ssm = unsafe {
        Tensor::empty_(
            (mixer.heads, mixer.head_dim, mixer.state_dim),
            DType::F32,
            &mixer.device,
        )?
    };
    let params = Params {
        tokens: as_i32(tokens)?,
        heads: as_i32(mixer.heads)?,
        head_dim: as_i32(mixer.head_dim)?,
        groups: as_i32(mixer.groups)?,
        state_dim: as_i32(mixer.state_dim)?,
        kernel: as_i32(mixer.kernel)?,
        conv_dim: as_i32(mixer.conv_dim)?,
        in_dim: as_i32(in_dim)?,
        projected_dtype: dtype_code(projected.dtype())?,
        output_dtype: dtype_code(mixer.dtype)?,
        dt_min: mixer.dt_limit.0 as f32,
        dt_max: mixer.dt_limit.1 as f32,
        eps: mixer.eps as f32,
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
    if chunked_scan && tokens >= CHUNK_TOKENS * 2 {
        let chunks = tokens.div_ceil(CHUNK_TOKENS);
        let alphas = unsafe { Tensor::empty_((chunks, mixer.heads), DType::F32, &mixer.device)? };
        // The prefix kernel replaces each summary with that chunk's starting
        // state after reading its summary. The scan reads those starts later.
        let summaries = unsafe {
            Tensor::empty_(
                (chunks, mixer.heads, mixer.head_dim, mixer.state_dim),
                DType::F32,
                &mixer.device,
            )?
        };
        let summary = device
            .get_func(&module, CHUNK_SUMMARY)
            .ok_or_else(|| candle_core::Error::Msg("missing Mamba chunk summary kernel".into()))?;
        unsafe {
            summary.launch(
                LaunchConfig {
                    grid_dim: (as_u32(mixer.heads)?, as_u32(chunks)?, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    pointer(&conv)?,
                    pointer(&dt_decay)?,
                    pointer(&alphas)?,
                    pointer(&summaries)?,
                    as_i32(CHUNK_TOKENS)?,
                    params,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Mamba chunk summary: {e}")))?;
        let prefix = device
            .get_func(&module, CHUNK_PREFIX)
            .ok_or_else(|| candle_core::Error::Msg("missing Mamba chunk prefix kernel".into()))?;
        unsafe {
            prefix.launch(
                LaunchConfig {
                    grid_dim: (as_u32(mixer.heads)?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    pointer(&state.ssm)?,
                    pointer(&alphas)?,
                    pointer(&summaries)?,
                    as_i32(chunks)?,
                    params,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Mamba chunk prefix: {e}")))?;
        let scan = device
            .get_func(&module, CHUNK_SCAN)
            .ok_or_else(|| candle_core::Error::Msg("missing Mamba chunk scan kernel".into()))?;
        unsafe {
            scan.launch(
                LaunchConfig {
                    grid_dim: (as_u32(mixer.heads)?, as_u32(chunks)?, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    pointer(&conv)?,
                    pointer(&dt_decay)?,
                    pointer(&summaries)?,
                    pointer(&mixer.d)?,
                    pointer(&y)?,
                    pointer(&next_ssm)?,
                    as_i32(CHUNK_TOKENS)?,
                    params,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Mamba chunk scan: {e}")))?;
    } else {
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
    }
    // The subsequent Candle operations and async frees use this CUDA stream.
    // Ordering is preserved without blocking the host once per Mamba layer.
    if *SYNC_SCAN {
        device
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("synchronize Mamba scan: {e}")))?;
    }
    state.conv = next_conv;
    state.ssm = next_ssm;

    let group_rows = inner / mixer.groups;
    let threads = group_rows.min(1024).next_power_of_two();
    let shared_bytes = (group_rows + 32) * std::mem::size_of::<f32>();
    if fused_gate_norm && group_rows.is_multiple_of(4) && shared_bytes <= 48 * 1024 {
        let normalized = unsafe { Tensor::empty_((tokens, inner), mixer.dtype, &mixer.device)? };
        let kernel = device
            .get_func(&module, GATE_NORM)
            .ok_or_else(|| candle_core::Error::Msg("missing Mamba gate norm kernel".into()))?;
        unsafe {
            kernel.launch(
                LaunchConfig {
                    grid_dim: (as_u32(mixer.groups)?, as_u32(tokens)?, 1),
                    block_dim: (as_u32(threads)?, 1, 1),
                    shared_mem_bytes: as_u32(shared_bytes)?,
                },
                (
                    pointer(&projected)?,
                    pointer(&y)?,
                    pointer(&mixer.norm_weight)?,
                    pointer(&normalized)?,
                    params,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Mamba gate norm: {e}")))?;
        return mixer.out_proj.forward(&normalized);
    }

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

/// One launch updates the carried conv/SSM state and writes the normalized
/// output of a single decode token. The projection remains a GEMV.
pub(super) fn run_step(
    mixer: &Mamba2,
    projected: &Tensor,
    state: &mut MambaState,
) -> Result<Tensor> {
    let inner = mixer.heads * mixer.head_dim;
    let in_dim = inner + mixer.conv_dim + mixer.heads;
    if !supported_step(mixer, projected)
        || projected.dims() != [in_dim]
        || state.conv.dims() != [mixer.kernel - 1, mixer.conv_dim]
        || state.ssm.dims() != [mixer.heads, mixer.head_dim, mixer.state_dim]
    {
        candle_core::bail!("unsupported Nemotron Mamba decode shape")
    }
    let module = module(mixer)?;
    let device = mixer.device.as_cuda_device()?.cuda_device();
    // The step kernel writes every group row of normalized, every conv
    // channel of next_conv, and every SSM state element before either is read.
    let normalized = unsafe { Tensor::empty_((1, inner), mixer.dtype, &mixer.device)? };
    let next_conv = unsafe {
        Tensor::empty_(
            (mixer.kernel - 1, mixer.conv_dim),
            DType::F32,
            &mixer.device,
        )?
    };
    let next_ssm = unsafe {
        Tensor::empty_(
            (mixer.heads, mixer.head_dim, mixer.state_dim),
            DType::F32,
            &mixer.device,
        )?
    };
    let params = Params {
        tokens: 1,
        heads: as_i32(mixer.heads)?,
        head_dim: as_i32(mixer.head_dim)?,
        groups: as_i32(mixer.groups)?,
        state_dim: as_i32(mixer.state_dim)?,
        kernel: as_i32(mixer.kernel)?,
        conv_dim: as_i32(mixer.conv_dim)?,
        in_dim: as_i32(in_dim)?,
        projected_dtype: dtype_code(projected.dtype())?,
        output_dtype: dtype_code(mixer.dtype)?,
        dt_min: mixer.dt_limit.0 as f32,
        dt_max: mixer.dt_limit.1 as f32,
        eps: mixer.eps as f32,
    };
    let group_rows = inner / mixer.groups;
    let shared_bytes = as_u32((256 + group_rows + 256) * std::mem::size_of::<f32>())?;
    let kernel = device
        .get_func(&module, STEP)
        .ok_or_else(|| candle_core::Error::Msg("missing Mamba decode kernel".into()))?;
    unsafe {
        kernel.launch(
            LaunchConfig {
                grid_dim: (as_u32(mixer.groups)?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            },
            (
                pointer(projected)?,
                pointer(&state.conv)?,
                pointer(&next_conv)?,
                pointer(&mixer.conv_weight)?,
                pointer(&mixer.conv_bias)?,
                pointer(&mixer.dt_bias)?,
                pointer(&mixer.a)?,
                pointer(&mixer.d)?,
                pointer(&mixer.norm_weight)?,
                pointer(&state.ssm)?,
                pointer(&next_ssm)?,
                pointer(&normalized)?,
                params,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Mamba decode: {e}")))?;
    state.conv = next_conv;
    state.ssm = next_ssm;
    mixer.out_proj.forward(&normalized)
}

#[cfg(test)]
mod tests;
