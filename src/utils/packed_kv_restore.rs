//! Validated dense packed KV pages and batched expansion into Flash BF16 slots.
//! This wire format is separate from TurboQuant and from yesno bitplanes.

use crate::utils::GpuKvCache;
use bincode::Options;
use candle_core::{DType, Result};
use ring::digest::{Context, SHA256};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"XPKV\0\0\0\x01";
pub const DENSE_PACKED_KV_VERSION: u32 = 1;
const MAX_PAGE_WORDS: usize = 1 << 26;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum PackedKvAxis {
    /// Each head/channel vector is grouped along the old-token axis.
    Key,
    /// Each old token and head is grouped along the channel axis.
    Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedKvException {
    /// Token-major output index within the page, before the exact tail.
    pub index: u32,
    pub bf16_bits: u16,
}

/// All integers on the wire use bincode fixed-width little-endian encoding.
/// `codes` are LSB-first 2/4-bit words; `params` interleave f32 min and step.
/// The exact tail is head/channel-major for K, token-major for V.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensePackedKvPage {
    pub version: u32,
    pub axis: PackedKvAxis,
    pub bits: u8,
    pub tokens: u32,
    pub heads: u32,
    pub channels: u32,
    pub group_size: u32,
    pub tail_tokens: u32,
    #[serde(with = "crate::models::layers::state_bytes")]
    pub codes: Vec<u8>,
    pub params: Vec<f32>,
    pub tail_bf16: Vec<u16>,
    pub exceptions: Vec<PackedKvException>,
    pub sha256: [u8; 32],
}

impl DensePackedKvPage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        axis: PackedKvAxis,
        bits: u8,
        tokens: u32,
        heads: u32,
        channels: u32,
        group_size: u32,
        tail_tokens: u32,
        codes: Vec<u8>,
        params: Vec<f32>,
        tail_bf16: Vec<u16>,
        exceptions: Vec<PackedKvException>,
    ) -> Result<Self> {
        let mut page = Self {
            version: DENSE_PACKED_KV_VERSION,
            axis,
            bits,
            tokens,
            heads,
            channels,
            group_size,
            tail_tokens,
            codes,
            params,
            tail_bf16,
            exceptions,
            sha256: [0; 32],
        };
        page.validate_shape()?;
        page.sha256 = page.digest();
        Ok(page)
    }

    fn digest(&self) -> [u8; 32] {
        let mut hash = Context::new(&SHA256);
        hash.update(b"xinfer-dense-packed-kv-v1");
        hash.update(&self.version.to_le_bytes());
        hash.update(&[
            match self.axis {
                PackedKvAxis::Key => 0,
                PackedKvAxis::Value => 1,
            },
            self.bits,
        ]);
        for value in [
            self.tokens,
            self.heads,
            self.channels,
            self.group_size,
            self.tail_tokens,
        ] {
            hash.update(&value.to_le_bytes());
        }
        for len in [
            self.codes.len(),
            self.params.len(),
            self.tail_bf16.len(),
            self.exceptions.len(),
        ] {
            hash.update(&(len as u64).to_le_bytes());
        }
        hash.update(&self.codes);
        #[cfg(target_endian = "little")]
        {
            hash.update(bytemuck::cast_slice(&self.params));
            hash.update(bytemuck::cast_slice(&self.tail_bf16));
        }
        #[cfg(not(target_endian = "little"))]
        {
            for &value in &self.params {
                hash.update(&value.to_bits().to_le_bytes());
            }
            for &value in &self.tail_bf16 {
                hash.update(&value.to_le_bytes());
            }
        }
        for exception in &self.exceptions {
            hash.update(&exception.index.to_le_bytes());
            hash.update(&exception.bf16_bits.to_le_bytes());
        }
        let digest = hash.finish();
        let mut result = [0; 32];
        result.copy_from_slice(digest.as_ref());
        result
    }

    fn validate_shape(&self) -> Result<()> {
        if self.version != DENSE_PACKED_KV_VERSION {
            candle_core::bail!("unsupported dense packed KV page version {}", self.version)
        }
        if !matches!(self.bits, 2 | 4)
            || self.tokens == 0
            || self.heads == 0
            || self.channels == 0
            || self.group_size == 0
            || self.tail_tokens > self.tokens
        {
            candle_core::bail!("invalid dense packed KV page geometry")
        }
        let tokens = self.tokens as usize;
        let heads = self.heads as usize;
        let channels = self.channels as usize;
        let old = tokens - self.tail_tokens as usize;
        let words = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| candle_core::Error::Msg("packed KV page size overflow".into()))?;
        if words > MAX_PAGE_WORDS {
            candle_core::bail!("dense packed KV page exceeds word limit")
        }
        let old_words = old * heads * channels;
        let code_bytes = (old_words * self.bits as usize).div_ceil(8);
        let group_size = self.group_size as usize;
        let groups_per_vector = match self.axis {
            PackedKvAxis::Key => old.div_ceil(group_size),
            PackedKvAxis::Value => channels.div_ceil(group_size),
        };
        let groups = match self.axis {
            PackedKvAxis::Key => heads * channels * groups_per_vector,
            PackedKvAxis::Value => old * heads * groups_per_vector,
        };
        if self.codes.len() != code_bytes
            || self.params.len() != groups * 2
            || self.tail_bf16.len() != self.tail_tokens as usize * heads * channels
        {
            candle_core::bail!("dense packed KV payload length mismatch")
        }
        if let Some(&last) = self.codes.last() {
            let used = old_words * self.bits as usize % 8;
            if used != 0 && last >> used != 0 {
                candle_core::bail!("nonzero dense packed KV code padding")
            }
        }
        let max_code = ((1u32 << self.bits) - 1) as f64;
        for pair in self.params.chunks_exact(2) {
            let high = pair[0] as f64 + max_code * pair[1] as f64;
            if !pair[0].is_finite()
                || !pair[1].is_finite()
                || pair[1] < 0.0
                || !(high as f32).is_finite()
            {
                candle_core::bail!("invalid dense packed KV affine parameter")
            }
        }
        let mut previous = None;
        for exception in &self.exceptions {
            if exception.index as usize >= old_words
                || previous.is_some_and(|index| index >= exception.index)
            {
                candle_core::bail!("invalid or duplicate dense packed KV exception")
            }
            previous = Some(exception.index);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_shape()?;
        if self.digest() != self.sha256 {
            candle_core::bail!("dense packed KV page SHA-256 mismatch")
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        self.serialize()
    }

    // Called only on pages returned directly by the GPU encoder. Their
    // constructor has already checked the shape and computed the checksum;
    // no caller can mutate a page between that constructor and this call.
    #[cfg(feature = "cuda")]
    fn into_fresh_bytes(self) -> Result<Vec<u8>> {
        self.serialize()
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = MAGIC.to_vec();
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut bytes, self)
            .map_err(|e| candle_core::Error::Msg(format!("serialize packed KV page: {e}")))?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload = bytes.strip_prefix(MAGIC).ok_or_else(|| {
            candle_core::Error::Msg("invalid dense packed KV envelope magic/version".into())
        })?;
        let page: Self = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(payload.len() as u64)
            .reject_trailing_bytes()
            .deserialize(payload)
            .map_err(|e| candle_core::Error::Msg(format!("deserialize packed KV page: {e}")))?;
        page.validate()?;
        Ok(page)
    }

    /// Portable byte-parity oracle. Output is token-major `[T,H,D]` BF16 bits.
    pub fn decode_cpu_bf16(&self) -> Result<Vec<u16>> {
        self.validate()?;
        let old = (self.tokens - self.tail_tokens) as usize;
        let heads = self.heads as usize;
        let channels = self.channels as usize;
        let tail = self.tail_tokens as usize;
        let mut output = vec![0u16; self.tokens as usize * heads * channels];
        for (index, word) in output.iter_mut().enumerate() {
            let token = index / (heads * channels);
            let hd = index % (heads * channels);
            if token >= old {
                let rank = match self.axis {
                    PackedKvAxis::Key => hd * tail + token - old,
                    PackedKvAxis::Value => (token - old) * heads * channels + hd,
                };
                *word = self.tail_bf16[rank];
                continue;
            }
            let rank = match self.axis {
                PackedKvAxis::Key => hd * old + token,
                PackedKvAxis::Value => index,
            };
            let bit = rank * self.bits as usize;
            let code = (self.codes[bit / 8] >> (bit % 8)) & ((1 << self.bits) - 1);
            let group = match self.axis {
                PackedKvAxis::Key => {
                    hd * old.div_ceil(self.group_size as usize) + token / self.group_size as usize
                }
                PackedKvAxis::Value => {
                    (token * heads + hd / channels) * channels.div_ceil(self.group_size as usize)
                        + (hd % channels) / self.group_size as usize
                }
            };
            let value = (self.params[2 * group] as f64
                + code as f64 * self.params[2 * group + 1] as f64) as f32;
            let bits = value.to_bits();
            *word = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
        }
        for exception in &self.exceptions {
            output[exception.index as usize] = exception.bf16_bits;
        }
        Ok(output)
    }
}

/// Encode token-major BF16 little-endian K or V bytes into individually
/// addressable 16-token dense pages. The most recent `exact_tail_tokens` stay
/// exact, including a partial final page. No peer or persistence type is used.
pub fn encode_dense_packed_kv_tile(
    bf16_le: &[u8],
    tokens: usize,
    heads: usize,
    channels: usize,
    axis: PackedKvAxis,
    bits: u8,
    exact_tail_tokens: usize,
) -> Result<Vec<DensePackedKvPage>> {
    const BLOCK: usize = 16;
    if !matches!(bits, 2 | 4) || tokens == 0 || heads == 0 || channels == 0 {
        candle_core::bail!("invalid dense packed KV tile geometry")
    }
    let width = heads
        .checked_mul(channels)
        .ok_or_else(|| candle_core::Error::Msg("KV width overflow".into()))?;
    let bytes = tokens
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| candle_core::Error::Msg("KV byte length overflow".into()))?;
    if bf16_le.len() != bytes {
        candle_core::bail!("BF16 KV tile byte length mismatch")
    }
    let exact_from = tokens.saturating_sub(exact_tail_tokens);
    let mut pages = Vec::with_capacity(tokens.div_ceil(BLOCK));
    for block in 0..tokens.div_ceil(BLOCK) {
        let first = block * BLOCK;
        let count = (tokens - first).min(BLOCK);
        let tail_tokens = (first + count).saturating_sub(exact_from.max(first));
        let old = count - tail_tokens;
        let page_bytes = &bf16_le[first * width * 2..(first + count) * width * 2];
        let word = |token: usize, head: usize, channel: usize| {
            let offset = (token * width + head * channels + channel) * 2;
            u16::from_le_bytes([page_bytes[offset], page_bytes[offset + 1]])
        };
        if !page_bytes.chunks_exact(2).all(|pair| {
            half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]]))
                .to_f32()
                .is_finite()
        }) {
            candle_core::bail!("non-finite BF16 KV value")
        }
        let code_len = old
            .checked_mul(width)
            .and_then(|n| n.checked_mul(bits as usize))
            .ok_or_else(|| candle_core::Error::Msg("KV code size overflow".into()))?
            .div_ceil(8);
        let mut codes = vec![0u8; code_len];
        let mut params = Vec::new();
        let mut tail = Vec::with_capacity(tail_tokens * width);
        let mut rank = 0usize;
        let mut encode_group = |values: &[f32]| {
            let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let step = ((maximum as f64 - minimum as f64) / ((1u32 << bits) - 1) as f64) as f32;
            params.extend_from_slice(&[minimum, step]);
            for &value in values {
                let code = if step == 0.0 {
                    0
                } else {
                    (((value as f64 - minimum as f64) / step as f64).round() as i64)
                        .clamp(0, ((1u32 << bits) - 1) as i64) as u8
                };
                let bit = rank * bits as usize;
                codes[bit / 8] |= code << (bit % 8);
                rank += 1;
            }
        };
        match axis {
            PackedKvAxis::Key => {
                for head in 0..heads {
                    for channel in 0..channels {
                        if old > 0 {
                            let mut values = [0f32; BLOCK];
                            for (token, value) in values.iter_mut().enumerate().take(old) {
                                *value = half::bf16::from_bits(word(token, head, channel)).to_f32();
                            }
                            encode_group(&values[..old]);
                        }
                        for token in old..count {
                            tail.push(word(token, head, channel));
                        }
                    }
                }
            }
            PackedKvAxis::Value => {
                let mut values = vec![0f32; channels];
                for token in 0..count {
                    for head in 0..heads {
                        if token < old {
                            for (channel, value) in values.iter_mut().enumerate() {
                                *value = half::bf16::from_bits(word(token, head, channel)).to_f32();
                            }
                            encode_group(&values);
                        } else {
                            for channel in 0..channels {
                                tail.push(word(token, head, channel));
                            }
                        }
                    }
                }
            }
        }
        pages.push(DensePackedKvPage::new(
            axis,
            bits,
            count as u32,
            heads as u32,
            channels as u32,
            match axis {
                PackedKvAxis::Key => BLOCK as u32,
                PackedKvAxis::Value => channels as u32,
            },
            tail_tokens as u32,
            codes,
            params,
            tail,
            Vec::new(),
        )?);
    }
    Ok(pages)
}

/// Encode a live CUDA Flash KV tile without reading its quantized BF16 prefix
/// back to the host. The returned pages use the same versioned wire format as
/// [`encode_dense_packed_kv_tile`]. The exact tail is read back separately.
#[cfg(feature = "cuda")]
pub fn encode_dense_packed_kv_tile_gpu(
    tile: &candle_core::Tensor,
    tokens: usize,
    axis: PackedKvAxis,
    bits: u8,
    exact_tail_tokens: usize,
) -> Result<Vec<DensePackedKvPage>> {
    encode::encode(tile, tokens, axis, bits, exact_tail_tokens)
}

/// One live Flash K/V tile for a bounded batched GPU encode. Requests must
/// share geometry and a CUDA device. Each tile may choose its own exact tail.
#[cfg(feature = "cuda")]
pub struct DensePackedKvEncode<'a> {
    pub tile: &'a candle_core::Tensor,
    pub axis: PackedKvAxis,
    pub exact_tail_tokens: usize,
}

/// Encode live CUDA tiles with at most 16 tiles staged per GPU batch.
#[cfg(feature = "cuda")]
pub fn encode_dense_packed_kv_tiles_gpu(
    requests: &[DensePackedKvEncode<'_>],
    tokens: usize,
    bits: u8,
) -> Result<Vec<Vec<DensePackedKvPage>>> {
    encode_batch::encode(requests, tokens, bits)
}

/// Encode live CUDA tiles into validated page envelopes. This path serializes
/// pages immediately after construction, so it does not hash them twice.
#[cfg(feature = "cuda")]
pub fn encode_dense_packed_kv_tiles_gpu_bytes(
    requests: &[DensePackedKvEncode<'_>],
    tokens: usize,
    bits: u8,
) -> Result<Vec<Vec<Vec<u8>>>> {
    encode_batch::encode_bytes(requests, tokens, bits)
}

pub struct DensePackedKvRestore<'a> {
    pub layer: usize,
    pub block: usize,
    pub token_offset: usize,
    pub page: &'a DensePackedKvPage,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DensePackedKvRestoreStats {
    pub pages: usize,
    pub packed_bytes: usize,
    pub output_bytes: usize,
    pub kernel_launches: usize,
}

/// Validate the complete batch before touching GPU slots. Only the standard
/// Flash `[blocks,tokens,heads,channels]` BF16 cache is supported. The caller
/// owns block allocation and page-table insertion.
pub fn restore_dense_packed_kv_pages(
    cache: &GpuKvCache,
    requests: &[DensePackedKvRestore<'_>],
) -> Result<DensePackedKvRestoreStats> {
    let GpuKvCache::Flash(pairs) = cache else {
        candle_core::bail!("dense packed KV restore requires a standard Flash cache")
    };
    let mut ranges: HashMap<(usize, PackedKvAxis, usize), Vec<(usize, usize)>> = HashMap::new();
    let mut stats = DensePackedKvRestoreStats::default();
    let mut device = None;
    for request in requests {
        let page = request.page;
        page.validate()?;
        let (key, value) = pairs
            .get(request.layer)
            .ok_or_else(|| candle_core::Error::Msg("dense packed KV layer out of range".into()))?;
        let target = if page.axis == PackedKvAxis::Key {
            key
        } else {
            value
        };
        if target.dtype() != DType::BF16 || target.rank() != 4 {
            candle_core::bail!("dense packed KV restore requires rank-4 BF16 slots")
        }
        let (blocks, block_tokens, heads, channels) = target.dims4()?;
        if !target.layout().is_contiguous()
            || request.block >= blocks
            || heads != page.heads as usize
            || channels != page.channels as usize
            || request
                .token_offset
                .checked_add(page.tokens as usize)
                .is_none_or(|end| end > block_tokens)
        {
            candle_core::bail!("dense packed KV destination shape, range, or stride mismatch")
        }
        if let Some(first) = &device {
            if !target.device().same_device(first) {
                candle_core::bail!("dense packed KV destinations must share one device")
            }
            #[cfg(feature = "cuda")]
            if !std::sync::Arc::ptr_eq(
                &target.device().as_cuda_device()?.cuda_device(),
                &first.as_cuda_device()?.cuda_device(),
            ) {
                candle_core::bail!("dense packed KV destinations must share one CUDA stream")
            }
        } else {
            device = Some(target.device().clone());
        }
        let interval = (
            request.token_offset,
            request.token_offset + page.tokens as usize,
        );
        let entry = ranges
            .entry((request.layer, page.axis, request.block))
            .or_default();
        if entry
            .iter()
            .any(|&(start, end)| interval.0 < end && start < interval.1)
        {
            candle_core::bail!("overlapping dense packed KV destination pages")
        }
        entry.push(interval);
        stats.pages += 1;
        stats.packed_bytes += page.codes.len()
            + page.params.len() * 4
            + page.tail_bf16.len() * 2
            + page.exceptions.len() * 6;
        stats.output_bytes += page.tokens as usize * heads * channels * 2;
    }
    if requests.is_empty() {
        return Ok(stats);
    }
    #[cfg(feature = "cuda")]
    {
        cuda::expand(pairs, requests, &mut stats)?;
        Ok(stats)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        candle_core::bail!("dense packed KV GPU expansion requires the cuda feature")
    }
}

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
mod encode;

#[cfg(feature = "cuda")]
mod encode_batch;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_encoder_preserves_tail_and_rejects_bad_input() -> Result<()> {
        let words: Vec<u16> = (0..18 * 2 * 3)
            .map(|n| half::bf16::from_f32((n % 13) as f32 / 4.0).to_bits())
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        for axis in [PackedKvAxis::Key, PackedKvAxis::Value] {
            for bits in [2, 4] {
                let pages = encode_dense_packed_kv_tile(&bytes, 18, 2, 3, axis, bits, 18)?;
                assert_eq!(pages.len(), 2);
                assert_eq!(pages[0].tokens, 16);
                assert_eq!(pages[1].tokens, 2);
                assert_eq!(
                    pages
                        .iter()
                        .flat_map(|page| page.decode_cpu_bf16().unwrap())
                        .collect::<Vec<_>>(),
                    words
                );
                let pages = encode_dense_packed_kv_tile(&bytes, 18, 2, 3, axis, bits, 17)?;
                assert_eq!(pages[0].tail_tokens, 15);
                assert_eq!(pages[1].tail_tokens, 2);
                assert_eq!(pages[1].decode_cpu_bf16()?, words[16 * 2 * 3..]);
                for page in pages {
                    assert_eq!(
                        DensePackedKvPage::from_bytes(&page.to_bytes()?)?.sha256,
                        page.sha256
                    );
                }
            }
        }
        assert!(encode_dense_packed_kv_tile(
            &bytes[..bytes.len() - 1],
            18,
            2,
            3,
            PackedKvAxis::Key,
            2,
            0
        )
        .is_err());
        let mut nonfinite = bytes;
        nonfinite[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
        assert!(
            encode_dense_packed_kv_tile(&nonfinite, 18, 2, 3, PackedKvAxis::Value, 4, 0).is_err()
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_encoder_matches_cpu_page_bytes() -> Result<()> {
        use candle_core::{Device, Tensor};

        let device = Device::new_cuda(0)?;
        let heads = 3;
        let channels = 13;
        let tokens = 37;
        let words: Vec<half::bf16> = (0..3 * 16 * heads * channels)
            .map(|n| {
                let value = ((n * 37 % 503) as f32 - 251.0) / 29.0;
                half::bf16::from_f32(value)
            })
            .collect();
        let tile = Tensor::from_vec(words.clone(), (3, 16, heads, channels), &device)?;
        let bytes: Vec<u8> = words[..tokens * heads * channels]
            .iter()
            .flat_map(|word| word.to_bits().to_le_bytes())
            .collect();
        for axis in [PackedKvAxis::Key, PackedKvAxis::Value] {
            for bits in [2, 4] {
                for tail in [0, 1, 5, 16, 19, 37] {
                    let cpu = encode_dense_packed_kv_tile(
                        &bytes, tokens, heads, channels, axis, bits, tail,
                    )?;
                    let gpu = encode_dense_packed_kv_tile_gpu(&tile, tokens, axis, bits, tail)?;
                    assert_eq!(cpu.len(), gpu.len());
                    for (index, (expected, actual)) in cpu.iter().zip(&gpu).enumerate() {
                        assert_eq!(
                            expected.to_bytes()?,
                            actual.to_bytes()?,
                            "GPU page differs at axis {axis:?}, {bits} bits, tail {tail}, page {index}"
                        );
                    }
                }
            }
        }
        let requests = [
            DensePackedKvEncode {
                tile: &tile,
                axis: PackedKvAxis::Key,
                exact_tail_tokens: 19,
            },
            DensePackedKvEncode {
                tile: &tile,
                axis: PackedKvAxis::Value,
                exact_tail_tokens: 5,
            },
            DensePackedKvEncode {
                tile: &tile,
                axis: PackedKvAxis::Key,
                exact_tail_tokens: 37,
            },
        ];
        let batch = encode_dense_packed_kv_tiles_gpu(&requests, tokens, 4)?;
        for (request, actual) in requests.iter().zip(batch) {
            let expected = encode_dense_packed_kv_tile(
                &bytes,
                tokens,
                heads,
                channels,
                request.axis,
                4,
                request.exact_tail_tokens,
            )?;
            assert_eq!(
                expected
                    .iter()
                    .map(DensePackedKvPage::to_bytes)
                    .collect::<Result<Vec<_>>>()?,
                actual
                    .iter()
                    .map(DensePackedKvPage::to_bytes)
                    .collect::<Result<Vec<_>>>()?
            );
        }
        let serialized = encode_dense_packed_kv_tiles_gpu_bytes(&requests, tokens, 4)?;
        for (request, actual) in requests.iter().zip(serialized) {
            let expected = encode_dense_packed_kv_tile(
                &bytes,
                tokens,
                heads,
                channels,
                request.axis,
                4,
                request.exact_tail_tokens,
            )?;
            assert_eq!(
                expected
                    .iter()
                    .map(DensePackedKvPage::to_bytes)
                    .collect::<Result<Vec<_>>>()?,
                actual
            );
        }
        // A BF16 midpoint can round to a different 2-bit code if the
        // subtraction and division are narrowed to f32 before rounding.
        let midpoint = [-0.0159912109375, -0.0120849609375, -0.0081787109375];
        let edge_words: Vec<half::bf16> = midpoint
            .into_iter()
            .cycle()
            .take(16 * 3)
            .map(half::bf16::from_f32)
            .collect();
        let edge_tile = Tensor::from_vec(edge_words.clone(), (1, 16, 1, 3), &device)?;
        let edge_bytes: Vec<u8> = edge_words
            .iter()
            .flat_map(|word| word.to_bits().to_le_bytes())
            .collect();
        let edge_cpu =
            encode_dense_packed_kv_tile(&edge_bytes, 16, 1, 3, PackedKvAxis::Value, 2, 0)?;
        let edge_request = [DensePackedKvEncode {
            tile: &edge_tile,
            axis: PackedKvAxis::Value,
            exact_tail_tokens: 0,
        }];
        let edge_gpu = encode_dense_packed_kv_tiles_gpu(&edge_request, 16, 2)?;
        assert_eq!(edge_cpu[0].to_bytes()?, edge_gpu[0][0].to_bytes()?);
        let edge_serialized = encode_dense_packed_kv_tiles_gpu_bytes(&edge_request, 16, 2)?;
        assert_eq!(edge_cpu[0].to_bytes()?, edge_serialized[0][0]);
        Ok(())
    }

    #[test]
    fn envelope_rejects_corruption_shape_and_duplicate_exceptions() -> Result<()> {
        let page = DensePackedKvPage::new(
            PackedKvAxis::Value,
            2,
            2,
            1,
            4,
            2,
            1,
            vec![0x39],
            vec![0.0, 0.5, 1.0, 0.25],
            vec![1, 2, 3, 4],
            vec![PackedKvException {
                index: 0,
                bf16_bits: 0x3f80,
            }],
        )?;
        let bytes = page.to_bytes()?;
        assert_eq!(
            DensePackedKvPage::from_bytes(&bytes)?.decode_cpu_bf16()?,
            page.decode_cpu_bf16()?
        );
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(DensePackedKvPage::from_bytes(&corrupt).is_err());
        let mut wrong = page.clone();
        wrong.channels = 8;
        assert!(wrong.validate().is_err());
        let mut wrong = page;
        wrong.exceptions.push(wrong.exceptions[0]);
        assert!(wrong.validate().is_err());
        Ok(())
    }
}
