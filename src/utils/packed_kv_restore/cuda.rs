use super::{
    packed_payload_bytes, DensePackedKvRestore, DensePackedKvRestoreOptions,
    DensePackedKvRestoreStats, PackedKvAxis,
};
use candle_core::cuda_backend::cudarc::driver::{
    capture_status, result::memcpy_htod_async, sys::CUstreamCaptureStatus, DevicePtr, DeviceRepr,
    LaunchAsync, LaunchConfig,
};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{Result, Storage, Tensor};
use std::collections::BTreeMap;
use std::sync::Mutex;

const MODULE: &str = "xinfer_dense_packed_kv_v2";
const KERNEL: &str = "expand_dense_packed_kv";
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
// The C layout is 8 bytes followed by fourteen 4-byte integers: no padding.
// The CUDA declaration in expand.cu has the same field order and size.
const _: () = assert!(std::mem::size_of::<GpuPage>() == 64);
unsafe impl bytemuck::Zeroable for GpuPage {}
unsafe impl bytemuck::Pod for GpuPage {}

#[derive(Clone, Copy)]
struct ArenaLayout {
    codes: usize,
    params: usize,
    tails: usize,
    exception_indices: usize,
    exception_values: usize,
    bytes: usize,
}

fn aligned(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|end| end & !(alignment - 1))
        .ok_or_else(|| candle_core::Error::Msg("dense packed KV arena size overflow".into()))
}

fn append_bytes(offset: usize, items: usize, item_bytes: usize) -> Result<usize> {
    offset
        .checked_add(
            items.checked_mul(item_bytes).ok_or_else(|| {
                candle_core::Error::Msg("dense packed KV arena size overflow".into())
            })?,
        )
        .ok_or_else(|| candle_core::Error::Msg("dense packed KV arena size overflow".into()))
}

impl ArenaLayout {
    fn new(
        pages: usize,
        codes: usize,
        params: usize,
        tails: usize,
        exceptions: usize,
    ) -> Result<Self> {
        let codes_offset = append_bytes(0, pages, std::mem::size_of::<GpuPage>())?;
        let params_offset = aligned(append_bytes(codes_offset, codes, 1)?, 4)?;
        let tails_offset = aligned(append_bytes(params_offset, params, 4)?, 2)?;
        let indices_offset = aligned(append_bytes(tails_offset, tails, 2)?, 4)?;
        let values_offset = aligned(append_bytes(indices_offset, exceptions, 4)?, 2)?;
        let bytes = append_bytes(values_offset, exceptions, 2)?;
        for offset in [
            codes_offset,
            params_offset,
            tails_offset,
            indices_offset,
            values_offset,
            bytes,
        ] {
            as_u32(offset)?;
        }
        Ok(Self {
            codes: codes_offset,
            params: params_offset,
            tails: tails_offset,
            exception_indices: indices_offset,
            exception_values: values_offset,
            bytes,
        })
    }
}

struct BatchPlan {
    indices: Vec<usize>,
    descriptors: Vec<GpuPage>,
    layout: ArenaLayout,
    payload_bytes: usize,
    max_words: usize,
}

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
    options: DensePackedKvRestoreOptions,
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
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (index, request) in requests.iter().enumerate() {
        let words = request.page.tokens as usize
            * request.page.heads as usize
            * request.page.channels as usize;
        let bucket = (usize::BITS - 1 - words.leading_zeros()) / 3;
        groups.entry(bucket).or_default().push(index);
    }
    // Plan every batch before the first write. Besides preserving the hard
    // payload budget, this gives one finite arena capacity for the whole call.
    let mut plans = Vec::new();
    for group in groups.values() {
        let mut cursor = 0usize;
        while cursor < group.len() {
            let start = cursor;
            let mut payload_bytes = 0usize;
            let mut codes = 0usize;
            let mut params = 0usize;
            let mut tails = 0usize;
            let mut exceptions = 0usize;
            let mut max_words = 0usize;
            while cursor < group.len() && cursor - start < options.max_batch_pages {
                let page = requests[group[cursor]].page;
                let bytes = packed_payload_bytes(page)?;
                // Every page was checked against this hard limit before GPU
                // writes. Keep the guard here for the accumulated batch too.
                if bytes > options.max_batch_payload_bytes - payload_bytes {
                    break;
                }
                payload_bytes += bytes;
                codes = codes.checked_add(page.codes.len()).ok_or_else(|| {
                    candle_core::Error::Msg("dense packed KV code count overflow".into())
                })?;
                params = params.checked_add(page.params.len()).ok_or_else(|| {
                    candle_core::Error::Msg("dense packed KV parameter count overflow".into())
                })?;
                tails = tails.checked_add(page.tail_bf16.len()).ok_or_else(|| {
                    candle_core::Error::Msg("dense packed KV tail count overflow".into())
                })?;
                exceptions = exceptions
                    .checked_add(page.exceptions.len())
                    .ok_or_else(|| {
                        candle_core::Error::Msg("dense packed KV exception count overflow".into())
                    })?;
                max_words = max_words
                    .max(page.tokens as usize * page.heads as usize * page.channels as usize);
                cursor += 1;
            }
            if cursor == start {
                candle_core::bail!("dense packed KV page exceeds max_batch_payload_bytes")
            }
            let layout = ArenaLayout::new(cursor - start, codes, params, tails, exceptions)?;
            // Descriptor and at most seven alignment bytes are outside the
            // caller's packed-payload budget, but still bounded by page cap.
            let overhead_limit = options
                .max_batch_pages
                .checked_mul(std::mem::size_of::<GpuPage>())
                .and_then(|bytes| bytes.checked_add(7))
                .ok_or_else(|| {
                    candle_core::Error::Msg("dense packed KV arena overhead overflow".into())
                })?;
            if layout.bytes > options.max_batch_payload_bytes + overhead_limit {
                candle_core::bail!("dense packed KV aligned staging exceeds bounded arena")
            }
            let mut descriptors = Vec::with_capacity(cursor - start);
            let mut code_offset = 0usize;
            let mut param_offset = 0usize;
            let mut tail_offset = 0usize;
            let mut exception_offset = 0usize;
            for &index in &group[start..cursor] {
                let request = &requests[index];
                let page = request.page;
                let target = if page.axis == PackedKvAxis::Key {
                    &pairs[request.layer].0
                } else {
                    &pairs[request.layer].1
                };
                descriptors.push(GpuPage {
                    destination: destination_pointer(target, request.block)?,
                    code_offset: as_u32(code_offset)?,
                    param_offset: as_u32(param_offset)?,
                    tail_offset: as_u32(tail_offset)?,
                    exception_offset: as_u32(exception_offset)?,
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
                code_offset += page.codes.len();
                param_offset += page.params.len();
                tail_offset += page.tail_bf16.len();
                exception_offset += page.exceptions.len();
            }
            plans.push(BatchPlan {
                indices: group[start..cursor].to_vec(),
                descriptors,
                layout,
                payload_bytes,
                max_words,
            });
        }
    }
    let capacity = plans.iter().map(|plan| plan.layout.bytes).max().unwrap();
    // One page-locked host arena and one CUDA allocation are reused only after
    // the matching upload and kernel have completed on this stream.
    let mut host = unsafe { device.alloc_pinned::<u8>(capacity) }
        .map_err(|e| candle_core::Error::Msg(format!("allocate packed KV pinned arena: {e}")))?;
    stats.staging_host_allocations = 1;
    let arena_gpu = unsafe { device.alloc::<u8>(capacity) }
        .map_err(|e| candle_core::Error::Msg(format!("allocate packed KV GPU arena: {e}")))?;
    stats.staging_device_allocations = 1;
    let function = device
        .get_func(MODULE, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing dense packed KV kernel".into()))?;
    for plan in plans {
        let staging = &mut host.as_mut_slice()[..plan.layout.bytes];
        // Pinned memory starts uninitialized. This also initializes alignment
        // padding, which is uploaded but never read by the kernel.
        staging.fill(0);
        let mut code_offset = 0usize;
        let mut param_offset = 0usize;
        let mut tail_offset = 0usize;
        let mut exception_offset = 0usize;
        for index in &plan.indices {
            let request = &requests[*index];
            let page = request.page;
            let start = plan.layout.codes + code_offset;
            staging[start..start + page.codes.len()].copy_from_slice(&page.codes);
            let start = plan.layout.params + param_offset * 4;
            staging[start..start + page.params.len() * 4]
                .copy_from_slice(bytemuck::cast_slice(&page.params));
            let start = plan.layout.tails + tail_offset * 2;
            staging[start..start + page.tail_bf16.len() * 2]
                .copy_from_slice(bytemuck::cast_slice(&page.tail_bf16));
            for exception in &page.exceptions {
                let index = plan.layout.exception_indices + exception_offset * 4;
                staging[index..index + 4].copy_from_slice(&exception.index.to_le_bytes());
                let value = plan.layout.exception_values + exception_offset * 2;
                staging[value..value + 2].copy_from_slice(&exception.bf16_bits.to_le_bytes());
                exception_offset += 1;
            }
            code_offset += page.codes.len();
            param_offset += page.params.len();
            tail_offset += page.tail_bf16.len();
        }
        staging[..plan.descriptors.len() * std::mem::size_of::<GpuPage>()]
            .copy_from_slice(bytemuck::cast_slice(&plan.descriptors));
        let blocks = as_u32(plan.max_words.div_ceil(256))?;
        let config = LaunchConfig {
            grid_dim: (blocks, as_u32(plan.indices.len())?, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        device
            .bind_to_thread()
            .map_err(|e| candle_core::Error::Msg(format!("bind packed KV CUDA context: {e}")))?;
        // The upload and kernel share one stream. Synchronize even on a
        // submission error before reusing or dropping the pinned buffer.
        let submitted =
            unsafe { memcpy_htod_async(*arena_gpu.device_ptr(), staging, *device.cu_stream()) }
                .map_err(|e| candle_core::Error::Msg(format!("upload packed KV arena: {e}")))
                .and_then(|_| unsafe {
                    function
                        .clone()
                        .launch(
                            config,
                            (
                                &arena_gpu,
                                as_u32(plan.layout.codes)?,
                                as_u32(plan.layout.params)?,
                                as_u32(plan.layout.tails)?,
                                as_u32(plan.layout.exception_indices)?,
                                as_u32(plan.layout.exception_values)?,
                            ),
                        )
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("launch dense packed KV: {e}"))
                        })
                });
        let synchronized = device
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("synchronize dense packed KV: {e}")));
        submitted?;
        synchronized?;
        stats.kernel_launches += 1;
        stats.staging_uploads += 1;
        stats.batch_synchronizations += 1;
        stats.max_batch_pages = stats.max_batch_pages.max(plan.indices.len());
        stats.max_batch_payload_bytes = stats.max_batch_payload_bytes.max(plan.payload_bytes);
        stats.max_batch_staging_bytes = stats.max_batch_staging_bytes.max(plan.layout.bytes);
    }
    Ok(())
}
