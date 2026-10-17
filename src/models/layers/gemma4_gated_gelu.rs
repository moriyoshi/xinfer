//! Opt-in fused BF16 GELU-and-multiply for Gemma 4's merged NVFP4 MLP.

use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::Mutex;

const KERNEL: &str = "xinfer_gemma4_gated_gelu_bf16";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

fn pointer(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Gemma gated GELU requires CUDA tensors")
    };
    let base = *storage.as_cuda_slice::<half::bf16>()?.device_ptr();
    Ok(base + (layout.start_offset() * DType::BF16.size_in_bytes()) as u64)
}

fn module(device: &candle_core::CudaDevice) -> Result<String> {
    let cuda = device.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(device).ok_or_else(|| {
        candle_core::Error::Msg("unknown CUDA architecture for Gemma gated GELU".into())
    })?;
    if sm < 80 {
        candle_core::bail!("Gemma gated GELU requires compute capability 8.0 or later")
    }
    let name = format!("xinfer_gemma4_gated_gelu_sm{sm}_v1");
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !cuda.has_func(&name, KERNEL) {
        let capture = capture_status(*cuda.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check gated GELU capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Gemma gated GELU must be loaded before CUDA graph capture")
        }
        let ptx = compile_ptx_with_opts(
            include_str!("gemma4_gated_gelu.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile gated GELU: {e}")))?;
        cuda.load_ptx(ptx, &name, &[KERNEL])
            .map_err(|e| candle_core::Error::Msg(format!("load gated GELU: {e}")))?;
    }
    Ok(name)
}

pub(crate) fn prepare(device: &candle_core::Device) -> Result<()> {
    module(device.as_cuda_device()?)?;
    Ok(())
}

/// Returns None for layouts or dtypes outside the fused kernel's contract.
pub(crate) fn forward(gate_up: &Tensor, width: usize) -> Result<Option<Tensor>> {
    if gate_up.dtype() != DType::BF16
        || !gate_up.device().is_cuda()
        || !gate_up.is_contiguous()
        || gate_up.rank() != 2
        || width == 0
        || width > i32::MAX as usize
        || gate_up.dim(1)? != width * 2
    {
        return Ok(None);
    }
    let tokens = gate_up.dim(0)?;
    let elements = tokens
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("Gemma gated GELU element count overflow".into()))?;
    if elements == 0 {
        return Ok(None);
    }
    let device = gate_up.device().as_cuda_device()?;
    let cuda = device.cuda_device();
    let name = module(device)?;
    // Every output element is written by exactly one thread on the same CUDA
    // stream as its later consumer; uninitialized storage is never observed.
    let output = unsafe { Tensor::empty_((tokens, width), DType::BF16, gate_up.device())? };
    let function = cuda
        .get_func(&name, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing Gemma gated GELU kernel".into()))?;
    unsafe {
        function.launch(
            LaunchConfig {
                grid_dim: (elements.div_ceil(256).min(65_535) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer(gate_up)?,
                pointer(&output)?,
                elements as u64,
                width as i32,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Gemma gated GELU: {e}")))?;
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_gated_gelu_matches_separate_bf16_ops() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0) < 80 {
            return Ok(());
        }
        for (tokens, width) in [(3, 17), (128, 256), (32, 10_240)] {
            let values = (0..tokens * width * 2)
                .map(|i| ((i * 97 % 211) as f32 - 105.0) / 41.0)
                .collect::<Vec<_>>();
            let gate_up =
                Tensor::from_vec(values, (tokens, width * 2), &device)?.to_dtype(DType::BF16)?;
            let gate = gate_up.narrow(1, 0, width)?;
            let up = gate_up.narrow(1, width, width)?;
            let expected = (gate.gelu()? * up)?;
            let actual = forward(&gate_up, width)?.expect("contiguous BF16 gate/up");
            device.synchronize()?;
            let expected = expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = actual
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_abs = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 0.03125,
                "tokens={tokens} width={width} max_abs={max_abs}"
            );
        }
        Ok(())
    }
}
