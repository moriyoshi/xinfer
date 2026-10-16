//! Opt-in fused BF16 residual addition and FP32 RMSNorm for Nemotron-H prefill.

use crate::models::layers::others::NormX;
use candle_core::cuda_backend::cudarc::driver::{
    capture_status, sys::CUstreamCaptureStatus, DevicePtr, LaunchAsync, LaunchConfig,
};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::{LazyLock, Mutex};

const KERNEL: &str = "nemotron_residual_rmsnorm";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_FUSED_RESIDUAL_NORM").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});

pub(super) fn enabled() -> bool {
    *ENABLED
}

fn pointer(tensor: &Tensor, dtype: DType) -> Result<u64> {
    if tensor.dtype() != dtype || !tensor.is_contiguous() {
        candle_core::bail!("Nemotron fused norm needs contiguous {dtype:?} tensors")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Nemotron fused norm needs CUDA tensors")
    };
    let (base, bytes) = match dtype {
        DType::BF16 => (*storage.as_cuda_slice::<half::bf16>()?.device_ptr(), 2usize),
        DType::F32 => (*storage.as_cuda_slice::<f32>()?.device_ptr(), 4),
        _ => candle_core::bail!("unsupported Nemotron fused norm dtype {dtype:?}"),
    };
    let offset = layout
        .start_offset()
        .checked_mul(bytes)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron fused norm offset overflow".into()))?;
    base.checked_add(offset as u64)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron fused norm pointer overflow".into()))
}

pub(super) fn forward(base: &Tensor, update: &Tensor, norm: &NormX) -> Result<(Tensor, Tensor)> {
    let (weight, eps) = norm
        .f32_rms_weight_eps()
        .ok_or_else(|| candle_core::Error::Msg("Nemotron fused norm needs FP32 RMSNorm".into()))?;
    forward_with_params(base, update, &weight, eps)
}

fn forward_with_params(
    base: &Tensor,
    update: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (rows, hidden) = base.dims2()?;
    if update.dims() != base.dims()
        || base.dtype() != DType::BF16
        || update.dtype() != DType::BF16
        || !base.is_contiguous()
        || !update.is_contiguous()
        || !base.device().is_cuda()
        || hidden == 0
    {
        candle_core::bail!("unsupported Nemotron fused residual norm input")
    }
    if weight.dims() != [hidden] || weight.dtype() != DType::F32 {
        candle_core::bail!("Nemotron fused norm weight shape or dtype mismatch")
    }
    let rows = u32::try_from(rows)
        .map_err(|_| candle_core::Error::Msg("Nemotron fused norm rows exceed u32".into()))?;
    let hidden = i32::try_from(hidden)
        .map_err(|_| candle_core::Error::Msg("Nemotron fused norm width exceeds i32".into()))?;
    let cuda = base.device().as_cuda_device()?.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(base.device().as_cuda_device()?)
        .ok_or_else(|| candle_core::Error::Msg("unknown Nemotron CUDA capability".into()))?;
    let module = format!("xinfer_nemotron_residual_norm_sm{sm}_v1");
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !cuda.has_func(&module, KERNEL) {
            if capture_status(*cuda.cu_stream()).map_err(|e| {
                candle_core::Error::Msg(format!("check Nemotron norm CUDA capture: {e}"))
            })? == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
            {
                candle_core::bail!("Nemotron fused norm must be loaded before graph capture")
            }
            let ptx = compile_ptx_with_opts(
                include_str!("residual_norm.cu"),
                CompileOptions {
                    options: vec![format!("--gpu-architecture=compute_{sm}")],
                    ..Default::default()
                },
            )
            .map_err(|e| candle_core::Error::Msg(format!("compile Nemotron fused norm: {e}")))?;
            cuda.load_ptx(ptx, &module, &[KERNEL])
                .map_err(|e| candle_core::Error::Msg(format!("load Nemotron fused norm: {e}")))?;
        }
    }
    let residual = unsafe { Tensor::empty_(base.dims(), DType::BF16, base.device())? };
    let normalized = unsafe { Tensor::empty_(base.dims(), DType::BF16, base.device())? };
    let func = cuda
        .get_func(&module, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing Nemotron fused norm kernel".into()))?;
    unsafe {
        func.launch(
            LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (if hidden < 1024 { 32 } else { 1024 }, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer(base, DType::BF16)?,
                pointer(update, DType::BF16)?,
                pointer(weight, DType::F32)?,
                pointer(&residual, DType::BF16)?,
                pointer(&normalized, DType::BF16)?,
                hidden,
                eps,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Nemotron fused norm: {e}")))?;
    Ok((residual, normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Module;

    #[test]
    fn fused_residual_is_exact_and_norm_tracks_fp32_reference() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        for (rows, hidden) in [(3, 128), (7, 2688), (2, 4480)] {
            let a = Tensor::from_vec(
                (0..rows * hidden)
                    .map(|i| ((i * 17 % 277) as f32 - 138.0) / 43.0)
                    .collect::<Vec<_>>(),
                (rows, hidden),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let b = Tensor::from_vec(
                (0..rows * hidden)
                    .map(|i| ((i * 29 % 251) as f32 - 125.0) / 71.0)
                    .collect::<Vec<_>>(),
                (rows, hidden),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..hidden)
                    .map(|i| 0.75 + (i % 23) as f32 / 53.0)
                    .collect::<Vec<_>>(),
                hidden,
                &device,
            )?;
            let (residual, normalized) = forward_with_params(&a, &b, &weight, 1e-5)?;
            let expected_residual = (&a + &b)?;
            let bits = |tensor: &Tensor| -> Result<Vec<u16>> {
                Ok(tensor
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect())
            };
            assert_eq!(bits(&residual)?, bits(&expected_residual)?);
            let expected = candle_nn::RmsNorm::new(weight.clone(), 1e-5)
                .forward(&expected_residual.to_dtype(DType::F32)?)?
                .to_dtype(DType::BF16)?;
            assert_eq!(bits(&normalized)?, bits(&expected)?, "{rows}x{hidden}");
        }
        Ok(())
    }
}
