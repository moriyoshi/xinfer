//! Opt-in BF16 GEMV for Gemma 4 single-token decode on CUDA.

use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::cell::Cell;
use std::sync::Mutex;

const KERNEL: &str = "xinfer_gemma4_bf16_gemv";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct Guard(bool);

impl Drop for Guard {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(self.0));
    }
}

pub(crate) fn activate(enabled: bool) -> Guard {
    let previous = ACTIVE.with(|active| active.replace(enabled));
    Guard(previous)
}

pub(crate) fn active() -> bool {
    ACTIVE.with(Cell::get)
}

fn pointer(tensor: &Tensor) -> Result<u64> {
    if tensor.dtype() != DType::BF16 || !tensor.is_contiguous() {
        candle_core::bail!("Gemma 4 GEMV requires contiguous BF16 tensors")
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Gemma 4 GEMV requires CUDA tensors")
    };
    let base = *storage.as_cuda_slice::<half::bf16>()?.device_ptr();
    Ok(base + (layout.start_offset() * DType::BF16.size_in_bytes()) as u64)
}

fn module(device: &candle_core::CudaDevice) -> Result<String> {
    let cuda = device.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(device).ok_or_else(|| {
        candle_core::Error::Msg("unknown CUDA architecture for Gemma GEMV".into())
    })?;
    if sm < 80 {
        candle_core::bail!("Gemma 4 BF16 GEMV requires CUDA compute capability 8.0 or later")
    }
    let name = format!("xinfer_gemma4_bf16_gemv_sm{sm}_v1");
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !cuda.has_func(&name, KERNEL) {
        let capture = capture_status(*cuda.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Gemma GEMV capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Gemma 4 GEMV must be loaded before CUDA graph capture")
        }
        let ptx = compile_ptx_with_opts(
            include_str!("gemma4_gemv.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile Gemma GEMV: {e}")))?;
        cuda.load_ptx(ptx, &name, &[KERNEL])
            .map_err(|e| candle_core::Error::Msg(format!("load Gemma GEMV: {e}")))?;
    }
    Ok(name)
}

pub(crate) fn prepare(device: &candle_core::Device) -> Result<()> {
    module(device.as_cuda_device()?)?;
    Ok(())
}

pub(crate) fn forward(input: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
    if !active()
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || weight.rank() != 2
        || input.rank() == 0
    {
        return Ok(None);
    }
    let rows = weight.dim(0)?;
    let columns = weight.dim(1)?;
    if rows == 0
        || columns == 0
        || columns % 2 != 0
        || rows > i32::MAX as usize
        || columns > i32::MAX as usize
        || input.elem_count() != columns
        || input.dim(input.rank() - 1)? != columns
    {
        return Ok(None);
    }
    let weight_ptr = pointer(weight)?;
    let input_ptr = pointer(input)?;
    // The kernel reads two BF16 values through each 32-bit load. A narrowed
    // contiguous view can start at an odd BF16 offset, so retain cuBLAS there.
    if weight_ptr % 4 != 0 || input_ptr % 4 != 0 {
        return Ok(None);
    }
    let device = input.device().as_cuda_device()?;
    let cuda = device.cuda_device();
    let module = module(device)?;
    let mut output_shape = input.dims().to_vec();
    *output_shape.last_mut().unwrap() = rows;
    // The kernel writes every row before the result is read on this stream.
    let output = unsafe { Tensor::empty_(output_shape, DType::BF16, input.device())? };
    let function = cuda
        .get_func(&module, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing Gemma GEMV CUDA kernel".into()))?;
    unsafe {
        function.launch(
            LaunchConfig {
                grid_dim: (rows.div_ceil(8) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                weight_ptr,
                input_ptr,
                pointer(&output)?,
                rows as i32,
                columns as i32,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Gemma GEMV: {e}")))?;
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_gemv_matches_candle_and_writes_tail_rows() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0) < 80 {
            return Ok(());
        }
        for (rows, columns) in [(17, 64), (1033, 2560)] {
            let input = Tensor::from_vec(
                (0..columns)
                    .map(|i| ((i * 37 % 127) as f32 - 63.0) / 512.0)
                    .collect::<Vec<_>>(),
                (1, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..rows * columns)
                    .map(|i| ((i * 19 % 131) as f32 - 65.0) / 1024.0)
                    .collect::<Vec<_>>(),
                (rows, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let expected = input.matmul(&weight.t()?)?;
            let _guard = activate(true);
            let actual = forward(&input, &weight)?.expect("supported BF16 GEMV");
            device.synchronize()?;
            assert_eq!(actual.dims(), &[1, rows]);
            let expected = expected
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let actual = actual
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let max_abs = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_abs <= 0.03125, "rows={rows} max_abs={max_abs}");
        }
        let misaligned = Tensor::zeros((1, 65), DType::BF16, &device)?.narrow(1, 1, 64)?;
        let weight = Tensor::zeros((17, 64), DType::BF16, &device)?;
        let _guard = activate(true);
        assert_eq!(pointer(&misaligned)? % 4, 2);
        assert!(forward(&misaligned, &weight)?.is_none());
        Ok(())
    }
}
