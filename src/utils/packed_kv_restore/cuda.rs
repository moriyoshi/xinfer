use super::{
    packed_payload_bytes, DensePackedKvRestore, DensePackedKvRestoreExecution,
    DensePackedKvRestoreOptions, DensePackedKvRestorePhaseTimings, DensePackedKvRestoreStats,
    PackedKvAxis,
};
use candle_core::cuda_backend::cudarc::driver::{
    capture_status,
    result::{event, memcpy_htod_async},
    sys::{self, CUstreamCaptureStatus},
    DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{Result, Storage, Tensor};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

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

    // The five payload sections and the descriptor section are written in
    // full by stage_batch. Only alignment gaps need explicit initialization.
    fn zero_padding(
        self,
        staging: &mut [u8],
        descriptors: usize,
        codes: usize,
        params: usize,
        tails: usize,
        exceptions: usize,
    ) -> Result<()> {
        let ends = [
            append_bytes(0, descriptors, std::mem::size_of::<GpuPage>())?,
            append_bytes(self.codes, codes, 1)?,
            append_bytes(self.params, params, 4)?,
            append_bytes(self.tails, tails, 2)?,
            append_bytes(self.exception_indices, exceptions, 4)?,
            append_bytes(self.exception_values, exceptions, 2)?,
        ];
        let starts = [
            self.codes,
            self.params,
            self.tails,
            self.exception_indices,
            self.exception_values,
            self.bytes,
        ];
        if staging.len() != self.bytes || ends.iter().zip(starts).any(|(&end, start)| end > start) {
            candle_core::bail!("dense packed KV staging layout mismatch")
        }
        for (&end, start) in ends.iter().zip(starts) {
            staging[end..start].fill(0);
        }
        Ok(())
    }
}

struct BatchPlan {
    indices: Vec<usize>,
    descriptors: Vec<GpuPage>,
    layout: ArenaLayout,
    payload_bytes: usize,
    max_words: usize,
}

struct StagedLengths {
    codes: usize,
    params: usize,
    tails: usize,
    exceptions: usize,
}

fn stage_batch(
    staging: &mut [u8],
    plan: &BatchPlan,
    requests: &[DensePackedKvRestore<'_>],
) -> StagedLengths {
    let mut code_offset = 0usize;
    let mut param_offset = 0usize;
    let mut tail_offset = 0usize;
    let mut exception_offset = 0usize;
    for &index in &plan.indices {
        let page = requests[index].page;
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
    StagedLengths {
        codes: code_offset,
        params: param_offset,
        tails: tail_offset,
        exceptions: exception_offset,
    }
}

struct CompletionEvent(sys::CUevent);

impl CompletionEvent {
    fn new() -> Result<Self> {
        event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map(Self)
            .map_err(|e| candle_core::Error::Msg(format!("create packed KV completion event: {e}")))
    }
}

impl Drop for CompletionEvent {
    fn drop(&mut self) {
        // CUDA permits event destruction before completion; on a drain error
        // the arenas themselves are retained until process teardown.
        let _ = unsafe { event::destroy(self.0) };
    }
}

fn elapsed_ns(start: Option<Instant>) -> u64 {
    start.map_or(0, |start| start.elapsed().as_nanos() as u64)
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
    execution: DensePackedKvRestoreExecution,
    stats: &mut DensePackedKvRestoreStats,
    mut timings: Option<&mut DensePackedKvRestorePhaseTimings>,
) -> Result<()> {
    let module_start = timings.as_ref().map(|_| Instant::now());
    let first = &requests[0];
    let first_target = if first.page.axis == PackedKvAxis::Key {
        &pairs[first.layer].0
    } else {
        &pairs[first.layer].1
    };
    let candle_device = first_target.device();
    let device = candle_device.as_cuda_device()?.cuda_device();
    device
        .bind_to_thread()
        .map_err(|e| candle_core::Error::Msg(format!("bind packed KV CUDA context: {e}")))?;
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
    if let Some(timing) = timings.as_deref_mut() {
        timing.module_ns += elapsed_ns(module_start);
    }
    let planning_start = timings.as_ref().map(|_| Instant::now());
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
    let slots = if matches!(execution, DensePackedKvRestoreExecution::TwoBuffer { .. }) {
        plans.len().min(2)
    } else {
        1
    };
    let aggregate = capacity
        .checked_mul(slots)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| {
            candle_core::Error::Msg("dense packed KV aggregate staging size overflow".into())
        })?;
    if let DensePackedKvRestoreExecution::TwoBuffer {
        max_aggregate_staging_bytes,
    } = execution
    {
        if aggregate > max_aggregate_staging_bytes {
            candle_core::bail!("dense packed KV aggregate staging exceeds two-buffer budget")
        }
    }
    stats.max_aggregate_staging_bytes = aggregate;
    if let Some(timing) = timings.as_deref_mut() {
        timing.planning_ns += elapsed_ns(planning_start);
    }
    let allocation_start = timings.as_ref().map(|_| Instant::now());
    let mut hosts = Vec::with_capacity(slots);
    let mut arenas_gpu = Vec::with_capacity(slots);
    for _ in 0..slots {
        hosts.push(unsafe { device.alloc_pinned::<u8>(capacity) }.map_err(|e| {
            candle_core::Error::Msg(format!("allocate packed KV pinned arena: {e}"))
        })?);
        arenas_gpu.push(
            unsafe { device.alloc::<u8>(capacity) }.map_err(|e| {
                candle_core::Error::Msg(format!("allocate packed KV GPU arena: {e}"))
            })?,
        );
    }
    stats.staging_host_allocations = slots;
    stats.staging_device_allocations = slots;
    let events = if slots == 2 {
        (0..slots)
            .map(|_| CompletionEvent::new())
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut in_flight = vec![false; slots];
    let function = device
        .get_func(MODULE, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing dense packed KV kernel".into()))?;
    if let Some(timing) = timings.as_deref_mut() {
        timing.allocation_ns += elapsed_ns(allocation_start);
    }
    // The closure can fail after an async upload. The error-path drain below
    // keeps both pinned and GPU arenas live until that upload finishes.
    let submitted = (|| -> Result<()> {
        for (batch, plan) in plans.iter().enumerate() {
            let slot = batch % slots;
            if in_flight[slot] {
                let sync_start = timings.as_ref().map(|_| Instant::now());
                unsafe { event::synchronize(events[slot].0) }.map_err(|e| {
                    candle_core::Error::Msg(format!("wait packed KV completion: {e}"))
                })?;
                if let Some(timing) = timings.as_deref_mut() {
                    timing.synchronization_ns += elapsed_ns(sync_start);
                }
                stats.batch_synchronizations += 1;
                in_flight[slot] = false;
            }
            let staging = &mut hosts[slot].as_mut_slice()[..plan.layout.bytes];
            let fill_start = timings.as_ref().map(|_| Instant::now());
            let lengths = stage_batch(staging, plan, requests);
            if let Some(timing) = timings.as_deref_mut() {
                timing.host_fill_ns += elapsed_ns(fill_start);
            }
            let padding_start = timings.as_ref().map(|_| Instant::now());
            plan.layout.zero_padding(
                staging,
                plan.descriptors.len(),
                lengths.codes,
                lengths.params,
                lengths.tails,
                lengths.exceptions,
            )?;
            if let Some(timing) = timings.as_deref_mut() {
                timing.padding_ns += elapsed_ns(padding_start);
            }
            let blocks = as_u32(plan.max_words.div_ceil(256))?;
            let config = LaunchConfig {
                grid_dim: (blocks, as_u32(plan.indices.len())?, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            device.bind_to_thread().map_err(|e| {
                candle_core::Error::Msg(format!("bind packed KV CUDA context: {e}"))
            })?;
            let submit_start = timings.as_ref().map(|_| Instant::now());
            unsafe {
                memcpy_htod_async(*arenas_gpu[slot].device_ptr(), staging, *device.cu_stream())
            }
            .map_err(|e| candle_core::Error::Msg(format!("upload packed KV arena: {e}")))?;
            unsafe {
                function.clone().launch(
                    config,
                    (
                        &arenas_gpu[slot],
                        as_u32(plan.layout.codes)?,
                        as_u32(plan.layout.params)?,
                        as_u32(plan.layout.tails)?,
                        as_u32(plan.layout.exception_indices)?,
                        as_u32(plan.layout.exception_values)?,
                    ),
                )
            }
            .map_err(|e| candle_core::Error::Msg(format!("launch dense packed KV: {e}")))?;
            if slots == 2 {
                unsafe { event::record(events[slot].0, *device.cu_stream()) }.map_err(|e| {
                    candle_core::Error::Msg(format!("record packed KV completion: {e}"))
                })?;
                in_flight[slot] = true;
            }
            if let Some(timing) = timings.as_deref_mut() {
                timing.submission_ns += elapsed_ns(submit_start);
            }
            if slots == 1 {
                let sync_start = timings.as_ref().map(|_| Instant::now());
                device.synchronize().map_err(|e| {
                    candle_core::Error::Msg(format!("synchronize dense packed KV: {e}"))
                })?;
                if let Some(timing) = timings.as_deref_mut() {
                    timing.synchronization_ns += elapsed_ns(sync_start);
                }
                stats.batch_synchronizations += 1;
            }
            stats.kernel_launches += 1;
            stats.staging_uploads += 1;
            stats.max_batch_pages = stats.max_batch_pages.max(plan.indices.len());
            stats.max_batch_payload_bytes = stats.max_batch_payload_bytes.max(plan.payload_bytes);
            stats.max_batch_staging_bytes = stats.max_batch_staging_bytes.max(plan.layout.bytes);
        }
        Ok(())
    })();
    let drained = if slots == 2 || submitted.is_err() {
        let final_start = timings.as_ref().map(|_| Instant::now());
        let result = device
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("drain dense packed KV stream: {e}")));
        if let Some(timing) = timings {
            timing.synchronization_ns += elapsed_ns(final_start);
        }
        result
    } else {
        Ok(())
    };
    if slots == 2 && submitted.is_ok() {
        stats.batch_synchronizations += 1;
    }
    if drained.is_err() {
        // CUDA did not confirm completion. Keep both DMA source and GPU arena
        // alive rather than allowing their destructors to free in-flight data.
        std::mem::forget(hosts);
        std::mem::forget(arenas_gpu);
    }
    submitted?;
    drained
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::packed_kv_restore::{DensePackedKvPage, PackedKvException};

    #[test]
    fn aligned_arena_initializes_every_uploaded_byte() -> Result<()> {
        // The cases cover zero-length sections, alignment gaps, odd code
        // lengths, exceptions, and a smaller batch reusing a larger arena.
        let cases = [
            (1, 0, 0, 512, 0),
            (3, 17, 8, 0, 0),
            (2, 7, 5, 13, 3),
            (1, 1, 0, 0, 0),
        ];
        let capacity = cases
            .iter()
            .map(|&(p, c, a, t, e)| ArenaLayout::new(p, c, a, t, e).unwrap().bytes)
            .max()
            .unwrap();
        let mut arena = vec![0xa5u8; capacity];
        for (pages, codes, params, tails, exceptions) in cases {
            let layout = ArenaLayout::new(pages, codes, params, tails, exceptions)?;
            let slice = &mut arena[..layout.bytes];
            let ranges = [
                (0, pages * std::mem::size_of::<GpuPage>()),
                (layout.codes, layout.codes + codes),
                (layout.params, layout.params + params * 4),
                (layout.tails, layout.tails + tails * 2),
                (
                    layout.exception_indices,
                    layout.exception_indices + exceptions * 4,
                ),
                (
                    layout.exception_values,
                    layout.exception_values + exceptions * 2,
                ),
            ];
            for &(start, end) in &ranges {
                slice[start..end].fill(0x5a);
            }
            layout.zero_padding(slice, pages, codes, params, tails, exceptions)?;
            assert!(slice.iter().all(|&byte| byte != 0xa5));
            assert!(layout
                .zero_padding(slice, pages, codes + 100, params, tails, exceptions)
                .is_err());
            arena.fill(0xa5);
        }
        Ok(())
    }

    #[test]
    fn real_stage_writer_covers_exact_and_exception_sections() -> Result<()> {
        let exact = DensePackedKvPage::new(
            PackedKvAxis::Key,
            2,
            1,
            1,
            1,
            1,
            1,
            vec![],
            vec![],
            vec![0x3f80],
            vec![],
        )?;
        let quantized = DensePackedKvPage::new(
            PackedKvAxis::Key,
            2,
            3,
            1,
            1,
            2,
            0,
            vec![0x24],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![],
            vec![PackedKvException {
                index: 1,
                bf16_bits: 0x3f80,
            }],
        )?;
        let requests = [
            DensePackedKvRestore {
                layer: 0,
                block: 0,
                token_offset: 0,
                page: &exact,
            },
            DensePackedKvRestore {
                layer: 0,
                block: 1,
                token_offset: 0,
                page: &quantized,
            },
        ];
        let plan = BatchPlan {
            indices: vec![0, 1],
            descriptors: vec![<GpuPage as bytemuck::Zeroable>::zeroed(); 2],
            layout: ArenaLayout::new(2, 1, 4, 1, 1)?,
            payload_bytes: packed_payload_bytes(&exact)? + packed_payload_bytes(&quantized)?,
            max_words: 3,
        };
        let mut staging = vec![0xcdu8; plan.layout.bytes];
        let lengths = stage_batch(&mut staging, &plan, &requests);
        plan.layout.zero_padding(
            &mut staging,
            plan.descriptors.len(),
            lengths.codes,
            lengths.params,
            lengths.tails,
            lengths.exceptions,
        )?;
        assert!(staging.iter().all(|&byte| byte != 0xcd));
        assert_eq!(&staging[plan.layout.codes..plan.layout.codes + 1], &[0x24]);
        Ok(())
    }
}
