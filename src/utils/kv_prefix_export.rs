//! Export token-major BF16 Flash K/V prefixes as little-endian tile bytes.
//! Tiles are returned in layer K, V order. The caller owns model identity and
//! persistence metadata; this module never depends on a peer or snapshot type.

use std::sync::OnceLock;

use candle_core::{DType, Result, Storage, Tensor};
use half::bf16;
use rayon::prelude::*;

const HOST_BATCH_BYTES: usize = 32 * 1024 * 1024;
const MAX_BATCH_TILES: usize = 8;
static CONVERSION_POOL: OnceLock<std::result::Result<rayon::ThreadPool, String>> = OnceLock::new();

fn conversion_pool() -> Result<&'static rayon::ThreadPool> {
    CONVERSION_POOL
        .get_or_init(|| {
            let workers = std::thread::available_parallelism()
                .map_or(1, |available| available.get().min(MAX_BATCH_TILES));
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|index| format!("bf16-kv-export-{index}"))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(format!("BF16 KV export workers: {error}")))
}

fn geometry(cache: &[(Tensor, Tensor)], tokens: usize) -> Result<(usize, usize, usize)> {
    if cache.is_empty() || tokens == 0 {
        candle_core::bail!("empty BF16 KV cache or prefix")
    }
    let shape = cache[0].0.dims();
    if shape.len() != 4 || shape.contains(&0) {
        candle_core::bail!("expected nonempty rank-four Flash KV tiles")
    }
    let slots = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| candle_core::Error::Msg("KV slot count overflow".into()))?;
    if tokens > slots {
        candle_core::bail!("BF16 KV prefix exceeds allocated slots")
    }
    for (key, value) in cache {
        for tile in [key, value] {
            if tile.dims() != shape || tile.dtype() != DType::BF16 {
                candle_core::bail!("BF16 KV tile shape or dtype mismatch")
            }
            if tile.device().location() != cache[0].0.device().location() {
                candle_core::bail!("BF16 KV tiles must share a device")
            }
        }
    }
    let words = tokens
        .checked_mul(shape[2])
        .and_then(|x| x.checked_mul(shape[3]))
        .ok_or_else(|| candle_core::Error::Msg("KV prefix size overflow".into()))?;
    Ok((
        slots,
        words,
        words
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("KV prefix byte count overflow".into()))?,
    ))
}

fn little_endian_bytes(words: &[bf16]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice(words).to_vec()
    }
    #[cfg(not(target_endian = "little"))]
    {
        words
            .iter()
            .flat_map(|word| word.to_bits().to_le_bytes())
            .collect()
    }
}

/// Exact portable oracle; copies only the live prefix when the allocated tile
/// has substantial unused capacity.
pub fn export_bf16_kv_prefix_tilewise(
    cache: &[(Tensor, Tensor)],
    tokens: usize,
) -> Result<Vec<Vec<u8>>> {
    let (slots, _, _) = geometry(cache, tokens)?;
    let mut output = Vec::with_capacity(cache.len() * 2);
    for (key, value) in cache {
        for tile in [key, value] {
            let prefix = tile
                .reshape((slots, tile.dim(2)?, tile.dim(3)?))?
                .narrow(0, 0, tokens)?;
            let compact = if slots - tokens >= slots.div_ceil(20) {
                prefix.copy()?
            } else {
                prefix.contiguous()?
            };
            output.push(little_endian_bytes(
                &compact.flatten_all()?.to_vec1::<bf16>()?,
            ));
        }
    }
    Ok(output)
}

/// Export Flash BF16 tiles using a bounded CUDA readback buffer. At most 32 MiB
/// and eight tiles are staged at once. Oversized single tiles use the tilewise
/// path, which remains the byte-parity oracle. Host byte conversion uses one
/// process-wide pool with at most eight workers. Returned bytes are caller-owned.
pub fn export_bf16_kv_prefix(cache: &[(Tensor, Tensor)], tokens: usize) -> Result<Vec<Vec<u8>>> {
    let (slots, words_per_tile, bytes_per_tile) = geometry(cache, tokens)?;
    if !cache[0].0.device().is_cuda() || bytes_per_tile > HOST_BATCH_BYTES {
        return export_bf16_kv_prefix_tilewise(cache, tokens);
    }
    let batch_tiles = (HOST_BATCH_BYTES / bytes_per_tile).clamp(1, MAX_BATCH_TILES);
    let tiles: Vec<_> = cache.iter().flat_map(|(key, value)| [key, value]).collect();
    let mut staging = vec![bf16::from_bits(0); batch_tiles * words_per_tile];
    let mut output = Vec::with_capacity(tiles.len());
    let pool = conversion_pool()?;
    for batch in tiles.chunks(batch_tiles) {
        for (index, tile) in batch.iter().enumerate() {
            let prefix = tile
                .reshape((slots, tile.dim(2)?, tile.dim(3)?))?
                .narrow(0, 0, tokens)?;
            let (storage, layout) = prefix.storage_and_layout();
            let (start, end) = layout
                .contiguous_offsets()
                .ok_or_else(|| candle_core::Error::Msg("noncontiguous BF16 KV prefix".into()))?;
            if end - start != words_per_tile {
                candle_core::bail!("invalid BF16 KV prefix extent")
            }
            let cuda = match &*storage {
                Storage::Cuda(cuda) => cuda,
                _ => candle_core::bail!("expected CUDA KV storage"),
            };
            let view = cuda.as_cuda_slice::<bf16>()?.slice(start..end);
            cuda.device
                .cuda_device()
                .dtoh_sync_copy_into(
                    &view,
                    &mut staging[index * words_per_tile..(index + 1) * words_per_tile],
                )
                .map_err(|error| {
                    candle_core::Error::Msg(format!("BF16 KV CUDA readback: {error}"))
                })?;
        }
        output.extend(pool.install(|| {
            staging[..batch.len() * words_per_tile]
                .par_chunks_exact(words_per_tile)
                .map(little_endian_bytes)
                .collect::<Vec<_>>()
        }));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cpu_prefix_order_and_boundary() -> Result<()> {
        let words: Vec<bf16> = (0..32).map(|n| bf16::from_f32(n as f32)).collect();
        let key = Tensor::from_vec(words.clone(), (2, 4, 2, 2), &Device::Cpu)?;
        let value = Tensor::from_vec(
            words.into_iter().rev().collect::<Vec<_>>(),
            (2, 4, 2, 2),
            &Device::Cpu,
        )?;
        let cache = vec![(key, value)];
        let output = export_bf16_kv_prefix(&cache, 5)?;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 5 * 2 * 2 * 2);
        assert_eq!(output, export_bf16_kv_prefix_tilewise(&cache, 5)?);
        assert!(export_bf16_kv_prefix(&cache, 9).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_prefix_matches_oracle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let words: Vec<bf16> = (0..(32 * 16 * 8))
            .map(|n| bf16::from_f32((n % 31) as f32))
            .collect();
        let tile = Tensor::from_vec(words, (32, 16, 2, 4), &device)?;
        let cache = vec![(tile.clone(), tile.clone()), (tile.clone(), tile)];
        for tokens in [8, 127, 128, 512] {
            assert_eq!(
                export_bf16_kv_prefix(&cache, tokens)?,
                export_bf16_kv_prefix_tilewise(&cache, tokens)?
            );
        }
        Ok(())
    }
}
