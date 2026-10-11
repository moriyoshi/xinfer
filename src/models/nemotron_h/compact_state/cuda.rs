use super::CompactMambaFrame;
use crate::models::nemotron_h::MambaState;
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{DType, Device, Result, Storage, Tensor};
use std::sync::Mutex;

const MODULE: &str = "xinfer_compact_mamba_v1";
const KERNEL: &str = "expand_compact_mamba";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn expand(
    frame: &CompactMambaFrame<'_>,
    device: &Device,
) -> Result<Vec<Option<MambaState>>> {
    let cuda_device = device.as_cuda_device()?.cuda_device();
    let capture = capture_status(*cuda_device.cu_stream())
        .map_err(|e| candle_core::Error::Msg(format!("check CUDA graph capture: {e}")))?;
    if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        candle_core::bail!("compact Mamba restore cannot run during CUDA graph capture")
    }
    // Do all compilation and validation before allocating the FP32 state.
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !cuda_device.has_func(MODULE, KERNEL) {
            let ptx = compile_ptx(include_str!("expand.cu")).map_err(|e| {
                candle_core::Error::Msg(format!("compile compact Mamba kernel: {e}"))
            })?;
            cuda_device
                .load_ptx(ptx, MODULE, &[KERNEL])
                .map_err(|e| candle_core::Error::Msg(format!("load compact Mamba PTX: {e}")))?;
        }
    }
    let layer_words = frame.conv_words + frame.heads * frame.values * frame.channels;
    let total_words = frame.layers * layer_words;
    let conv_total = frame.layers * frame.conv_words;
    let group_values = frame.groups.len() * frame.values;
    let work = conv_total + group_values;
    let as_u32 = |value: usize| -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("compact Mamba CUDA dimension exceeds u32".into()))
    };
    let output = Tensor::zeros((total_words,), DType::F32, device)?;
    let (storage, layout) = output.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("compact Mamba output must be CUDA")
    };
    let out_slice = storage.as_cuda_slice::<f32>()?;
    let out_ptr = (*out_slice.device_ptr()) + (layout.start_offset() * 4) as u64;
    let d_data = cuda_device
        .htod_sync_copy(frame.data)
        .map_err(|e| candle_core::Error::Msg(format!("upload compact Mamba bytes: {e}")))?;
    let d_groups = cuda_device
        .htod_sync_copy(&frame.groups)
        .map_err(|e| candle_core::Error::Msg(format!("upload compact Mamba groups: {e}")))?;
    let function = cuda_device
        .get_func(MODULE, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing compact Mamba kernel".into()))?;
    let config = LaunchConfig {
        grid_dim: (as_u32(work.div_ceil(256))?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        function.launch(
            config,
            (
                &d_data,
                &d_groups,
                out_ptr,
                as_u32(frame.conv_offset)?,
                as_u32(frame.conv_words)?,
                as_u32(frame.heads)?,
                as_u32(frame.values)?,
                as_u32(frame.channels)?,
                as_u32(frame.layers)?,
                as_u32(layer_words)?,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch compact Mamba kernel: {e}")))?;
    cuda_device
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("synchronize compact Mamba kernel: {e}")))?;
    let mut states = vec![None; frame.layout.decoder_layers as usize];
    let conv_shape = frame.layout.conv_shape.map(|v| v as usize);
    let ssm_shape = frame.layout.ssm_shape.map(|v| v as usize);
    for layer in 0..frame.layers {
        let layer_view = output.narrow(0, layer * layer_words, layer_words)?;
        let conv = layer_view
            .narrow(0, 0, frame.conv_words)?
            .reshape(&conv_shape)?;
        let ssm = layer_view
            .narrow(0, frame.conv_words, layer_words - frame.conv_words)?
            .reshape(&ssm_shape)?;
        states[frame.layout.model_layer_indices[layer] as usize] = Some(MambaState { conv, ssm });
    }
    Ok(states)
}
