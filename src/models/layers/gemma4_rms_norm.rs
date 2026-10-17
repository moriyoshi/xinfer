//! Direct BF16-input/FP32-weight RMSNorm for Gemma 4.

use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::Mutex;

const KERNEL: &str = "xinfer_gemma4_rms_norm_bf16_f32";
const ADD_KERNEL: &str = "xinfer_gemma4_rms_norm_add_bf16_f32";
const POST_PRE_KERNEL: &str = "xinfer_gemma4_post_attn_pre_ff_bf16";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

fn pointer_bf16(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Gemma RMSNorm requires CUDA tensors")
    };
    let base = *storage.as_cuda_slice::<half::bf16>()?.device_ptr();
    Ok(base + (layout.start_offset() * DType::BF16.size_in_bytes()) as u64)
}

fn pointer_f32(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Gemma RMSNorm requires CUDA tensors")
    };
    let base = *storage.as_cuda_slice::<f32>()?.device_ptr();
    Ok(base + (layout.start_offset() * DType::F32.size_in_bytes()) as u64)
}

fn module(device: &candle_core::CudaDevice) -> Result<String> {
    let cuda = device.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(device).ok_or_else(|| {
        candle_core::Error::Msg("unknown CUDA architecture for Gemma RMSNorm".into())
    })?;
    if sm < 80 {
        candle_core::bail!("Gemma RMSNorm requires compute capability 8.0 or later")
    }
    let name = format!("xinfer_gemma4_rms_norm_sm{sm}_v3");
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !cuda.has_func(&name, KERNEL) {
        let capture = capture_status(*cuda.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Gemma RMSNorm capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Gemma RMSNorm must be loaded before CUDA graph capture")
        }
        let ptx = compile_ptx_with_opts(
            include_str!("gemma4_rms_norm.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile Gemma RMSNorm: {e}")))?;
        cuda.load_ptx(ptx, &name, &[KERNEL, ADD_KERNEL, POST_PRE_KERNEL])
            .map_err(|e| candle_core::Error::Msg(format!("load Gemma RMSNorm: {e}")))?;
    }
    Ok(name)
}

pub(crate) fn prepare(device: &candle_core::Device) -> Result<()> {
    module(device.as_cuda_device()?)?;
    Ok(())
}

pub(crate) fn forward(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    forward_inner(input, weight, None, eps)
}

pub(crate) fn forward_add(
    input: &Tensor,
    weight: &Tensor,
    residual: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    forward_inner(input, weight, Some(residual), eps)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_attention_pre_feedforward(
    attention: &Tensor,
    residual: &Tensor,
    post_weight: &Tensor,
    pre_weight: &Tensor,
    post_eps: f32,
    pre_eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    if attention.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || post_weight.dtype() != DType::F32
        || pre_weight.dtype() != DType::F32
        || !attention.device().is_cuda()
        || !attention.device().same_device(residual.device())
        || !attention.device().same_device(post_weight.device())
        || !attention.device().same_device(pre_weight.device())
        || !attention.is_contiguous()
        || !residual.is_contiguous()
        || !post_weight.is_contiguous()
        || !pre_weight.is_contiguous()
        || attention.rank() != 2
        || attention.shape() != residual.shape()
    {
        return Ok(None);
    }
    let rows = attention.dim(0)?;
    let columns = attention.dim(1)?;
    if rows == 0
        || rows > u32::MAX as usize
        || columns == 0
        || columns > i32::MAX as usize
        || post_weight.dims() != [columns]
        || pre_weight.dims() != [columns]
    {
        return Ok(None);
    }
    let device = attention.device().as_cuda_device()?;
    let cuda = device.cuda_device();
    let name = module(device)?;
    let xs = unsafe { Tensor::empty_(attention.shape().clone(), DType::BF16, attention.device())? };
    let mlp_input =
        unsafe { Tensor::empty_(attention.shape().clone(), DType::BF16, attention.device())? };
    let function = cuda
        .get_func(&name, POST_PRE_KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing Gemma post-attention kernel".into()))?;
    unsafe {
        function.launch(
            LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (if columns < 1024 { 32 } else { 1024 }, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                pointer_bf16(attention)?,
                pointer_bf16(residual)?,
                pointer_f32(post_weight)?,
                pointer_f32(pre_weight)?,
                pointer_bf16(&xs)?,
                pointer_bf16(&mlp_input)?,
                columns as i32,
                post_eps,
                pre_eps,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Gemma post-attention: {e}")))?;
    Ok(Some((xs, mlp_input)))
}

fn forward_inner(
    input: &Tensor,
    weight: &Tensor,
    residual: Option<&Tensor>,
    eps: f32,
) -> Result<Option<Tensor>> {
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::F32
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || input.rank() != 2
        || weight.rank() != 1
    {
        return Ok(None);
    }
    let rows = input.dim(0)?;
    let columns = input.dim(1)?;
    if rows == 0
        || columns == 0
        || rows > u32::MAX as usize
        || columns > i32::MAX as usize
        || weight.dim(0)? != columns
    {
        return Ok(None);
    }
    if let Some(residual) = residual {
        if residual.dtype() != DType::BF16
            || !residual.device().same_device(input.device())
            || !residual.is_contiguous()
            || residual.shape() != input.shape()
        {
            return Ok(None);
        }
    }
    let device = input.device().as_cuda_device()?;
    let cuda = device.cuda_device();
    let name = module(device)?;
    let output = unsafe { Tensor::empty_(input.shape().clone(), DType::BF16, input.device())? };
    let function = cuda
        .get_func(
            &name,
            if residual.is_some() {
                ADD_KERNEL
            } else {
                KERNEL
            },
        )
        .ok_or_else(|| candle_core::Error::Msg("missing Gemma RMSNorm kernel".into()))?;
    let threads = if columns < 1024 { 32 } else { 1024 };
    let config = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    if let Some(residual) = residual {
        unsafe {
            function.launch(
                config,
                (
                    pointer_bf16(input)?,
                    pointer_f32(weight)?,
                    pointer_bf16(residual)?,
                    pointer_bf16(&output)?,
                    columns as i32,
                    eps,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Gemma RMSNorm+add: {e}")))?;
    } else {
        unsafe {
            function.launch(
                config,
                (
                    pointer_bf16(input)?,
                    pointer_f32(weight)?,
                    pointer_bf16(&output)?,
                    columns as i32,
                    eps,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch Gemma RMSNorm: {e}")))?;
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_bf16_rms_norm_matches_casted_f32_path() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0) < 80 {
            return Ok(());
        }
        for (rows, columns) in [(3, 96), (7, 1024), (128, 2560)] {
            let values = (0..rows * columns)
                .map(|i| ((i * 41 % 127) as f32 - 63.0) / 29.0)
                .collect::<Vec<_>>();
            let input =
                Tensor::from_vec(values, (rows, columns), &device)?.to_dtype(DType::BF16)?;
            let weights = (0..columns)
                .map(|i| 0.5 + (i % 31) as f32 / 43.0)
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(weights, columns, &device)?;
            let eps = 1e-6f32;
            let expected = candle_nn::ops::rms_norm(&input.to_dtype(DType::F32)?, &weight, eps)?
                .to_dtype(DType::BF16)?;
            let actual = forward(&input, &weight, eps)?.expect("contiguous BF16/F32 RMSNorm");
            let residual =
                Tensor::full(0.125f32, (rows, columns), &device)?.to_dtype(DType::BF16)?;
            let expected_add = (&expected + &residual)?;
            let actual_add =
                forward_add(&input, &weight, &residual, eps)?.expect("contiguous BF16 residual");
            let pre_weight = Tensor::from_vec(
                (0..columns)
                    .map(|i| 0.75 + (i % 23) as f32 / 53.0)
                    .collect::<Vec<_>>(),
                columns,
                &device,
            )?;
            let expected_pre =
                candle_nn::ops::rms_norm(&expected_add.to_dtype(DType::F32)?, &pre_weight, eps)?
                    .to_dtype(DType::BF16)?;
            let (paired_add, paired_pre) =
                post_attention_pre_feedforward(&input, &residual, &weight, &pre_weight, eps, eps)?
                    .expect("contiguous paired Gemma norms");
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
                "rows={rows} columns={columns} max_abs={max_abs}"
            );
            let expected_add = expected_add
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual_add = actual_add
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_add_abs = expected_add
                .iter()
                .zip(&actual_add)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_add_abs <= 0.03125,
                "rows={rows} columns={columns} max_add_abs={max_add_abs}"
            );
            let paired_add = paired_add
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let paired_pre = paired_pre
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected_pre = expected_pre
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_pair_add_abs = expected_add
                .iter()
                .zip(&paired_add)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_pair_pre_abs = expected_pre
                .iter()
                .zip(&paired_pre)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_pair_add_abs <= 0.03125 && max_pair_pre_abs <= 0.03125,
                "rows={rows} columns={columns} pair add={max_pair_add_abs} pre={max_pair_pre_abs}"
            );
        }
        Ok(())
    }
}
