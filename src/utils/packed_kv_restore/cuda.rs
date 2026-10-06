use super::{DensePackedKvRestore, DensePackedKvRestoreStats, PackedKvAxis};
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{Result, Storage, Tensor};
use std::collections::BTreeMap;
use std::sync::Mutex;

const MODULE: &str = "xinfer_dense_packed_kv_v1";
const KERNEL: &str = "expand_dense_packed_kv";
const MAX_BATCH_PAGES: usize = 128;
const MAX_BATCH_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuPage {
    destination: u64,
    code_offset: u32,
    param_offset: u32,
    tail_offset: u32,
    exception_offset: u32,
    exception_count: u32,
    tokens: u32,
    heads: u32,
    channels: u32,
    old_tokens: u32,
    tail_tokens: u32,
    bits: u32,
    group_size: u32,
    slot_offset: u32,
    key: u32,
}

// All fields are plain integers and the CUDA definition has the same C layout.
unsafe impl DeviceRepr for GpuPage {}

fn as_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("dense packed KV batch offset exceeds u32".into()))
}

fn destination_pointer(tensor: &Tensor, block: usize) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("dense packed KV destination must be CUDA")
    };
    let slice = storage.as_cuda_slice::<half::bf16>()?;
    let block_words = tensor.dims()[1..].iter().product::<usize>();
    let offset = layout
        .start_offset()
        .checked_add(block.checked_mul(block_words).ok_or_else(|| {
            candle_core::Error::Msg("dense packed KV block offset overflow".into())
        })?)
        .and_then(|words| words.checked_mul(2))
        .ok_or_else(|| candle_core::Error::Msg("dense packed KV byte offset overflow".into()))?;
    (*slice.device_ptr())
        .checked_add(offset as u64)
        .ok_or_else(|| candle_core::Error::Msg("dense packed KV device address overflow".into()))
}

pub(super) fn expand(
    pairs: &[(Tensor, Tensor)],
    requests: &[DensePackedKvRestore<'_>],
    stats: &mut DensePackedKvRestoreStats,
) -> Result<()> {
    let first = &requests[0];
    let first_target = if first.page.axis == PackedKvAxis::Key {
        &pairs[first.layer].0
    } else {
        &pairs[first.layer].1
    };
    let candle_device = first_target.device();
    let device = candle_device.as_cuda_device()?.cuda_device();
    let capture = capture_status(*device.cu_stream())
        .map_err(|e| candle_core::Error::Msg(format!("check CUDA graph capture: {e}")))?;
    if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        candle_core::bail!("dense packed KV restore cannot run during CUDA graph capture")
    }
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !device.has_func(MODULE, KERNEL) {
            let ptx = compile_ptx(include_str!("expand.cu"))
                .map_err(|e| candle_core::Error::Msg(format!("compile dense packed KV: {e}")))?;
            device
                .load_ptx(ptx, MODULE, &[KERNEL])
                .map_err(|e| candle_core::Error::Msg(format!("load dense packed KV PTX: {e}")))?;
        }
    }
    // Avoid launching a giant grid for a batch dominated by tiny pages.
    // A bucket spans at most an eightfold output-word range.
    let mut groups: BTreeMap<u32, Vec<&DensePackedKvRestore<'_>>> = BTreeMap::new();
    for request in requests {
        let words = request.page.tokens as usize
            * request.page.heads as usize
            * request.page.channels as usize;
        let bucket = (usize::BITS - 1 - words.leading_zeros()) / 3;
        groups.entry(bucket).or_default().push(request);
    }
    for group in groups.values() {
        let mut cursor = 0usize;
        while cursor < group.len() {
            let start = cursor;
            let mut payload_bytes = 0usize;
            while cursor < group.len() && cursor - start < MAX_BATCH_PAGES {
                let page = group[cursor].page;
                let bytes = page.codes.len()
                    + page.params.len() * 4
                    + page.tail_bf16.len() * 2
                    + page.exceptions.len() * 6;
                if cursor > start && payload_bytes + bytes > MAX_BATCH_PAYLOAD_BYTES {
                    break;
                }
                payload_bytes += bytes;
                cursor += 1;
            }
            let chunk = &group[start..cursor];
            let mut descriptors = Vec::with_capacity(chunk.len());
            let mut codes = Vec::new();
            let mut params = Vec::new();
            let mut tails = Vec::new();
            let mut exception_indices = Vec::new();
            let mut exception_values = Vec::new();
            let mut max_words = 0usize;
            for request in chunk {
                let page = request.page;
                for (current, additional) in [
                    (codes.len(), page.codes.len()),
                    (params.len(), page.params.len()),
                    (tails.len(), page.tail_bf16.len()),
                    (exception_indices.len(), page.exceptions.len()),
                ] {
                    as_u32(current.checked_add(additional).ok_or_else(|| {
                        candle_core::Error::Msg("dense packed KV batch length overflow".into())
                    })?)?;
                }
                let target = if page.axis == PackedKvAxis::Key {
                    &pairs[request.layer].0
                } else {
                    &pairs[request.layer].1
                };
                descriptors.push(GpuPage {
                    destination: destination_pointer(target, request.block)?,
                    code_offset: as_u32(codes.len())?,
                    param_offset: as_u32(params.len())?,
                    tail_offset: as_u32(tails.len())?,
                    exception_offset: as_u32(exception_indices.len())?,
                    exception_count: as_u32(page.exceptions.len())?,
                    tokens: page.tokens,
                    heads: page.heads,
                    channels: page.channels,
                    old_tokens: page.tokens - page.tail_tokens,
                    tail_tokens: page.tail_tokens,
                    bits: page.bits as u32,
                    group_size: page.group_size,
                    slot_offset: as_u32(request.token_offset)?,
                    key: u32::from(page.axis == PackedKvAxis::Key),
                });
                codes.extend_from_slice(&page.codes);
                params.extend_from_slice(&page.params);
                tails.extend_from_slice(&page.tail_bf16);
                for exception in &page.exceptions {
                    exception_indices.push(exception.index);
                    exception_values.push(exception.bf16_bits);
                }
                max_words = max_words
                    .max(page.tokens as usize * page.heads as usize * page.channels as usize);
            }
            // A batch may contain all-exact pages or no sparse exceptions. These
            // one-word placeholders are never dereferenced by the kernel.
            if codes.is_empty() {
                codes.push(0);
            }
            if params.is_empty() {
                params.push(0.0);
            }
            if tails.is_empty() {
                tails.push(0);
            }
            if exception_indices.is_empty() {
                exception_indices.push(0);
                exception_values.push(0);
            }
            let d_descriptors = device.htod_copy(descriptors).map_err(|e| {
                candle_core::Error::Msg(format!("upload packed KV descriptors: {e}"))
            })?;
            let d_codes = device
                .htod_copy(codes)
                .map_err(|e| candle_core::Error::Msg(format!("upload packed KV codes: {e}")))?;
            let d_params = device
                .htod_copy(params)
                .map_err(|e| candle_core::Error::Msg(format!("upload packed KV scales: {e}")))?;
            let d_tails = device
                .htod_copy(tails)
                .map_err(|e| candle_core::Error::Msg(format!("upload packed KV tail: {e}")))?;
            let d_indices = device.htod_copy(exception_indices).map_err(|e| {
                candle_core::Error::Msg(format!("upload packed KV exceptions: {e}"))
            })?;
            let d_values = device.htod_copy(exception_values).map_err(|e| {
                candle_core::Error::Msg(format!("upload packed KV exception values: {e}"))
            })?;
            let function = device
                .get_func(MODULE, KERNEL)
                .ok_or_else(|| candle_core::Error::Msg("missing dense packed KV kernel".into()))?;
            let blocks = as_u32(max_words.div_ceil(256))?;
            let config = LaunchConfig {
                grid_dim: (blocks, as_u32(chunk.len())?, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // The descriptors and all buffers live through synchronization below;
            // every validated destination tensor is held by `pairs` throughout.
            unsafe {
                function.launch(
                    config,
                    (
                        &d_descriptors,
                        &d_codes,
                        &d_params,
                        &d_tails,
                        &d_indices,
                        &d_values,
                    ),
                )
            }
            .map_err(|e| candle_core::Error::Msg(format!("launch dense packed KV: {e}")))?;
            device.synchronize().map_err(|e| {
                candle_core::Error::Msg(format!("synchronize dense packed KV: {e}"))
            })?;
            stats.kernel_launches += 1;
        }
    }
    Ok(())
}
