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

#[cfg(test)]
mod tests {
    use super::*;

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
