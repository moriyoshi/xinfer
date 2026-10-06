use super::{DensePackedKvEncode, DensePackedKvPage, PackedKvAxis};
use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx;
use candle_core::{DType, Result, Storage};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

const MODULE: &str = "xinfer_dense_packed_kv_encode_batch_v1";
const PARAM_KERNEL: &str = "dense_kv_params_batch";
const CODE_KERNEL: &str = "dense_kv_codes_batch";
const TAIL_KERNEL: &str = "dense_kv_tail_batch";
const MAX_BATCH_TILES: usize = 16;
const MAX_BATCH_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_RETAINED_HOST_BUFFERS: usize = 2;
static COMPILE_LOCK: Mutex<()> = Mutex::new(());
static PAGE_POOL: OnceLock<std::result::Result<rayon::ThreadPool, String>> = OnceLock::new();
static HOST_POOL: OnceLock<Mutex<Vec<HostScratch>>> = OnceLock::new();

#[derive(Default)]
struct HostScratch {
    params: Vec<f32>,
    codes: Vec<u8>,
    tails: Vec<u16>,
}

struct ScratchLease(HostScratch);

impl ScratchLease {
    fn acquire() -> Self {
        let pool = HOST_POOL.get_or_init(|| Mutex::new(Vec::new()));
        Self(pool.lock().unwrap().pop().unwrap_or_default())
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        let retained_bytes = self.0.params.capacity() * size_of::<f32>()
            + self.0.codes.capacity()
            + self.0.tails.capacity() * size_of::<u16>();
        if retained_bytes > MAX_BATCH_PAYLOAD_BYTES {
            return;
        }
        let pool = HOST_POOL.get_or_init(|| Mutex::new(Vec::new()));
        let mut pool = pool.lock().unwrap();
        if pool.len() < MAX_RETAINED_HOST_BUFFERS {
            pool.push(std::mem::take(&mut self.0));
        }
    }
}

fn page_pool() -> Result<&'static rayon::ThreadPool> {
    PAGE_POOL
        .get_or_init(|| {
            let workers =
                std::thread::available_parallelism().map_or(1, |available| available.get().min(8));
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|index| format!("gpu-kv-page-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(format!("GPU KV page pool: {error}")))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuEncodeTile {
    source: u64,
    exact_from: u32,
    key: u32,
}

// The CUDA declaration has the same C layout and contains only integers.
unsafe impl DeviceRepr for GpuEncodeTile {}

fn as_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("GPU KV encoder batch exceeds u32".into()))
}

pub(super) fn encode(
    requests: &[DensePackedKvEncode<'_>],
    tokens: usize,
    bits: u8,
) -> Result<Vec<Vec<DensePackedKvPage>>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(bits, 2 | 4) || tokens == 0 {
        candle_core::bail!("invalid GPU KV encoder bit width or prefix")
    }
    let (_, _, heads, channels) = requests[0].tile.dims4()?;
    let width = heads
        .checked_mul(channels)
        .ok_or_else(|| candle_core::Error::Msg("KV width overflow".into()))?;
    let pages = tokens.div_ceil(16);
    let max_groups = width.max(
        16usize
            .checked_mul(heads)
            .ok_or_else(|| candle_core::Error::Msg("KV parameter batch overflow".into()))?,
    );
    let param_bytes = pages
        .checked_mul(max_groups)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| candle_core::Error::Msg("KV parameter batch overflow".into()))?;
    let code_bytes = pages
        .checked_mul(16)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(bits as usize))
        .ok_or_else(|| candle_core::Error::Msg("KV code batch overflow".into()))?
        .div_ceil(8);
    let tail_bytes = requests
        .iter()
        .map(|request| request.exact_tail_tokens.min(tokens))
        .max()
        .unwrap_or(0)
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| candle_core::Error::Msg("KV tail batch overflow".into()))?;
    let per_tile_bytes = param_bytes
        .checked_add(code_bytes)
        .and_then(|n| n.checked_add(tail_bytes))
        .ok_or_else(|| candle_core::Error::Msg("KV GPU batch size overflow".into()))?;
    let batch_tiles = (MAX_BATCH_PAYLOAD_BYTES / per_tile_bytes.max(1)).clamp(1, MAX_BATCH_TILES);
    let mut scratch = ScratchLease::acquire();
    let mut output = Vec::with_capacity(requests.len());
    for batch in requests.chunks(batch_tiles) {
        output.extend(encode_chunk(batch, tokens, bits, &mut scratch.0)?);
    }
    Ok(output)
}

fn encode_chunk(
    requests: &[DensePackedKvEncode<'_>],
    tokens: usize,
    bits: u8,
    scratch: &mut HostScratch,
) -> Result<Vec<Vec<DensePackedKvPage>>> {
    const BLOCK: usize = 16;
    let first = requests[0].tile;
    if first.dtype() != DType::BF16 || first.rank() != 4 {
        candle_core::bail!("GPU KV batch needs rank-four BF16 Flash slots")
    }
    let (blocks, block_tokens, heads, channels) = first.dims4()?;
    let slots = blocks
        .checked_mul(block_tokens)
        .ok_or_else(|| candle_core::Error::Msg("KV slot count overflow".into()))?;
    let width = heads
        .checked_mul(channels)
        .ok_or_else(|| candle_core::Error::Msg("KV width overflow".into()))?;
    if heads == 0 || channels == 0 || tokens > slots {
        candle_core::bail!("invalid GPU KV batch geometry")
    }
    let source_words = slots
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("KV tile size overflow".into()))?;
    let pages = tokens.div_ceil(BLOCK);
    let param_stride = width
        .max(
            BLOCK
                .checked_mul(heads)
                .ok_or_else(|| candle_core::Error::Msg("KV parameter count overflow".into()))?,
        )
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::Msg("KV parameter count overflow".into()))?;
    let code_stride = BLOCK
        .checked_mul(width)
        .and_then(|n| n.checked_mul(bits as usize))
        .ok_or_else(|| candle_core::Error::Msg("KV code count overflow".into()))?
        .div_ceil(8);
    let tail_stride = requests
        .iter()
        .map(|request| request.exact_tail_tokens.min(tokens))
        .max()
        .unwrap_or(0)
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::Msg("KV exact-tail size overflow".into()))?;
    let tile_count = requests.len();
    let param_words = tile_count
        .checked_mul(pages)
        .and_then(|n| n.checked_mul(param_stride))
        .ok_or_else(|| candle_core::Error::Msg("KV parameter batch overflow".into()))?;
    let code_bytes = tile_count
        .checked_mul(pages)
        .and_then(|n| n.checked_mul(code_stride))
        .ok_or_else(|| candle_core::Error::Msg("KV code batch overflow".into()))?;
    let tail_words = tile_count
        .checked_mul(tail_stride)
        .ok_or_else(|| candle_core::Error::Msg("KV exact-tail batch overflow".into()))?;
    for value in [
        source_words,
        param_words,
        code_bytes,
        tail_words,
        pages,
        width,
    ] {
        as_u32(value)?;
    }
    let first_cuda = first.device().as_cuda_device()?;
    let device = first_cuda.cuda_device();
    let capture = capture_status(*device.cu_stream())
        .map_err(|e| candle_core::Error::Msg(format!("check CUDA graph capture: {e}")))?;
    if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        candle_core::bail!("GPU KV encoding cannot run during CUDA graph capture")
    }
    let mut descriptors = Vec::with_capacity(tile_count);
    for request in requests {
        let tile = request.tile;
        if tile.dtype() != DType::BF16 || tile.dims() != first.dims() {
            candle_core::bail!("GPU KV batch tile geometry mismatch")
        }
        let (storage, layout) = tile.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("GPU KV batch tile is not CUDA")
        };
        if !Arc::ptr_eq(&device, &storage.device.cuda_device()) {
            candle_core::bail!("GPU KV batch tiles must share one CUDA stream")
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("noncontiguous GPU KV tile".into()))?;
        if end - start != source_words {
            candle_core::bail!("invalid GPU KV batch tile extent")
        }
        let source = storage.as_cuda_slice::<half::bf16>()?.slice(start..end);
        descriptors.push(GpuEncodeTile {
            source: *source.device_ptr(),
            exact_from: as_u32(tokens.saturating_sub(request.exact_tail_tokens))?,
            key: u32::from(request.axis == PackedKvAxis::Key),
        });
    }
    {
        let _guard = COMPILE_LOCK.lock().unwrap();
        if !device.has_func(MODULE, PARAM_KERNEL) {
            let ptx = compile_ptx(include_str!("encode.cu"))
                .map_err(|e| candle_core::Error::Msg(format!("compile GPU KV batch: {e}")))?;
            device
                .load_ptx(ptx, MODULE, &[PARAM_KERNEL, CODE_KERNEL, TAIL_KERNEL])
                .map_err(|e| candle_core::Error::Msg(format!("load GPU KV batch: {e}")))?;
        }
    }
    let d_descriptors = device
        .htod_copy(descriptors.clone())
        .map_err(|e| candle_core::Error::Msg(format!("upload GPU KV batch descriptors: {e}")))?;
    let d_params = device
        .alloc_zeros::<f32>(param_words, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU KV batch parameters: {e}")))?;
    let d_codes = device
        .alloc_zeros::<u8>(code_bytes, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU KV batch codes: {e}")))?;
    let d_tails = device
        .alloc_zeros::<u16>(tail_words.max(1), false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU KV batch tails: {e}")))?;
    let d_invalid = device
        .alloc_zeros::<u32>(1, false)
        .map_err(|e| candle_core::Error::Msg(format!("allocate GPU KV batch status: {e}")))?;
    let params_fn = device
        .get_func(MODULE, PARAM_KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing GPU KV batch parameter kernel".into()))?;
    let max_groups = width.max(BLOCK * heads);
    let param_launch = LaunchConfig {
        grid_dim: (
            as_u32(max_groups.div_ceil(128))?,
            as_u32(pages)?,
            as_u32(tile_count)?,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        params_fn.launch(
            param_launch,
            (
                &d_descriptors,
                &d_params,
                &d_invalid,
                as_u32(tokens)?,
                as_u32(heads)?,
                as_u32(channels)?,
                bits as u32,
                as_u32(pages)?,
                as_u32(param_stride)?,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch GPU KV batch parameters: {e}")))?;
    let codes_fn = device
        .get_func(MODULE, CODE_KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing GPU KV batch code kernel".into()))?;
    let code_launch = LaunchConfig {
        grid_dim: (
            as_u32(code_stride.div_ceil(256))?,
            as_u32(pages)?,
            as_u32(tile_count)?,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        codes_fn.launch(
            code_launch,
            (
                &d_descriptors,
                &d_params,
                &d_codes,
                as_u32(tokens)?,
                as_u32(heads)?,
                as_u32(channels)?,
                bits as u32,
                as_u32(pages)?,
                as_u32(param_stride)?,
                as_u32(code_stride)?,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch GPU KV batch codes: {e}")))?;
    if tail_stride > 0 {
        let tail_fn = device
            .get_func(MODULE, TAIL_KERNEL)
            .ok_or_else(|| candle_core::Error::Msg("missing GPU KV batch tail kernel".into()))?;
        let tail_launch = LaunchConfig {
            grid_dim: (as_u32(tail_stride.div_ceil(256))?, as_u32(tile_count)?, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            tail_fn.launch(
                tail_launch,
                (
                    &d_descriptors,
                    &d_tails,
                    as_u32(tokens)?,
                    as_u32(width)?,
                    as_u32(tail_stride)?,
                ),
            )
        }
        .map_err(|e| candle_core::Error::Msg(format!("launch GPU KV batch tails: {e}")))?;
    }
    let invalid = device
        .dtoh_sync_copy(&d_invalid)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV batch status: {e}")))?;
    if invalid[0] != 0 {
        candle_core::bail!("non-finite BF16 KV value")
    }
    scratch.params.resize(param_words, 0.0);
    device
        .dtoh_sync_copy_into(&d_params, &mut scratch.params)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV batch parameters: {e}")))?;
    scratch.codes.resize(code_bytes, 0);
    device
        .dtoh_sync_copy_into(&d_codes, &mut scratch.codes)
        .map_err(|e| candle_core::Error::Msg(format!("read GPU KV batch codes: {e}")))?;
    scratch.tails.resize(tail_words, 0);
    if tail_stride > 0 {
        device
            .dtoh_sync_copy_into(&d_tails, &mut scratch.tails)
            .map_err(|e| candle_core::Error::Msg(format!("read GPU KV batch tails: {e}")))?;
    }
    if scratch
        .tails
        .iter()
        .any(|word| !half::bf16::from_bits(*word).to_f32().is_finite())
    {
        candle_core::bail!("non-finite BF16 KV value")
    }
    page_pool()?.install(|| {
        requests
            .par_iter()
            .enumerate()
            .map(|(tile, request)| {
                let exact_from = descriptors[tile].exact_from as usize;
                let mut tile_pages = Vec::with_capacity(pages);
                for page in 0..pages {
                    let first = page * BLOCK;
                    let count = (tokens - first).min(BLOCK);
                    let old = count.min(exact_from.saturating_sub(first));
                    let tail_count = count - old;
                    let code_len = (old * width * bits as usize).div_ceil(8);
                    let groups = if request.axis == PackedKvAxis::Key {
                        if old > 0 {
                            width
                        } else {
                            0
                        }
                    } else {
                        old * heads
                    };
                    let mut exact = Vec::with_capacity(tail_count * width);
                    if request.axis == PackedKvAxis::Key {
                        for hd in 0..width {
                            for token in old..count {
                                let rank = (first + token - exact_from) * width + hd;
                                exact.push(scratch.tails[tile * tail_stride + rank]);
                            }
                        }
                    } else {
                        for token in old..count {
                            let rank = tile * tail_stride + (first + token - exact_from) * width;
                            exact.extend_from_slice(&scratch.tails[rank..rank + width]);
                        }
                    }
                    let offset = (tile * pages + page) * code_stride;
                    let param_offset = (tile * pages + page) * param_stride;
                    tile_pages.push(DensePackedKvPage::new(
                        request.axis,
                        bits,
                        as_u32(count)?,
                        as_u32(heads)?,
                        as_u32(channels)?,
                        if request.axis == PackedKvAxis::Key {
                            16
                        } else {
                            as_u32(channels)?
                        },
                        as_u32(tail_count)?,
                        scratch.codes[offset..offset + code_len].to_vec(),
                        scratch.params[param_offset..param_offset + groups * 2].to_vec(),
                        exact,
                        Vec::new(),
                    )?);
                }
                Ok(tile_pages)
            })
            .collect::<Result<Vec<_>>>()
    })
}
