//! Portable, exact Gated DeltaNet state at a completed token boundary.
//!
//! Attention KV must be exported separately at the same boundary. The caller
//! supplies a SHA-256 model fingerprint covering weights, adapters, and
//! execution settings on both sides; xinfer cannot infer that identity from
//! an already loaded `VarBuilderX`.

use super::state_bytes;
use attention_rs::mamba_cache::MambaCache;
use bincode::Options;
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 8] = b"XGDN\0\0\0\x01";
pub const GDN_STATE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GdnStateDType {
    F32,
    F16,
    BF16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GdnStateLayout {
    /// Absolute decoder layer indices, in the order of the GDN cache.
    pub model_layer_indices: Vec<u32>,
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_world_size: u32,
    pub conv_dtype: GdnStateDType,
    pub recurrent_dtype: GdnStateDType,
    /// Per-sequence convolution state `[d_conv, kernel_size - 1]`.
    pub conv_shape: [u32; 2],
    /// Per-sequence recurrent state `[value_heads, key_dim, value_dim]`.
    pub recurrent_shape: [u32; 3],
}

impl GdnStateLayout {
    pub fn from_cache(
        cache: &MambaCache,
        model_layer_indices: Vec<u32>,
        tensor_parallel_rank: u32,
        tensor_parallel_world_size: u32,
    ) -> Result<Self> {
        if cache.num_gdn_layers() == 0 {
            candle_core::bail!("GDN state export requires at least one GDN layer")
        }
        let conv = cache.conv_state(0);
        let recurrent = cache.recurrent_state(0);
        let layout = Self {
            model_layer_indices,
            tensor_parallel_rank,
            tensor_parallel_world_size,
            conv_dtype: GdnStateDType::F32,
            recurrent_dtype: GdnStateDType::F32,
            conv_shape: [to_u32(conv.dim(1)?)?, to_u32(conv.dim(2)?)?],
            recurrent_shape: [
                to_u32(recurrent.dim(1)?)?,
                to_u32(recurrent.dim(2)?)?,
                to_u32(recurrent.dim(3)?)?,
            ],
        };
        layout.validate_cache(cache)?;
        Ok(layout)
    }

    fn validate_cache(&self, cache: &MambaCache) -> Result<()> {
        if self.tensor_parallel_world_size == 0
            || self.tensor_parallel_rank >= self.tensor_parallel_world_size
        {
            candle_core::bail!("invalid GDN tensor-parallel rank/world size")
        }
        if self.model_layer_indices.len() != cache.num_gdn_layers()
            || self
                .model_layer_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            candle_core::bail!("GDN layer count or order does not match the cache")
        }
        if self.conv_shape.contains(&0) || self.recurrent_shape.contains(&0) {
            candle_core::bail!("GDN state shapes must have nonzero dimensions")
        }
        if self.conv_dtype != GdnStateDType::F32 || self.recurrent_dtype != GdnStateDType::F32 {
            candle_core::bail!("GDN portable state requires FP32 tensors")
        }
        for layer in 0..cache.num_gdn_layers() {
            let conv = cache.conv_state(layer);
            let recurrent = cache.recurrent_state(layer);
            if conv.dtype() != DType::F32
                || recurrent.dtype() != DType::F32
                || conv.dims().get(1..) != Some(self.conv_shape.map(|v| v as usize).as_slice())
                || recurrent.dims().get(1..)
                    != Some(self.recurrent_shape.map(|v| v as usize).as_slice())
            {
                candle_core::bail!("GDN cache shape or dtype mismatch at layer {layer}")
            }
        }
        Ok(())
    }

    fn bytes_per_layer(&self) -> Result<usize> {
        let conv = checked_elements(&self.conv_shape)?;
        let recurrent = checked_elements(&self.recurrent_shape)?;
        conv.checked_add(recurrent)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("GDN state byte count overflow".into()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GdnStateSnapshot {
    pub version: u32,
    pub prefix_tokens: u64,
    /// Caller-supplied SHA-256 identity; the caller must derive it identically
    /// from weights, adapters, and numerical execution settings on both GPUs.
    pub model_fingerprint: [u8; 32],
    pub layout: GdnStateLayout,
    /// SHA-256 of `payload`.
    pub payload_sha256: [u8; 32],
    /// Contiguous little-endian FP32 bits: for each GDN layer, conv then recurrent.
    #[serde(with = "state_bytes")]
    pub payload: Vec<u8>,
}

impl GdnStateSnapshot {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate_payload()?;
        self.to_bytes_after_export()
    }

    // Safe only for the locally constructed snapshot before its public fields
    // can be mutated by a caller.
    fn to_bytes_after_export(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(
            MAGIC.len() + self.payload.len() + self.layout.model_layer_indices.len() * 4 + 256,
        );
        bytes.extend_from_slice(MAGIC);
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut bytes, self)
            .map_err(|e| candle_core::Error::Msg(format!("serialize GDN state: {e}")))?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let encoded = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| candle_core::Error::Msg("invalid GDN state magic/version".into()))?;
        let snapshot: Self = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(encoded.len() as u64)
            .reject_trailing_bytes()
            .deserialize(encoded)
            .map_err(|e| candle_core::Error::Msg(format!("deserialize GDN state: {e}")))?;
        snapshot.validate_payload()?;
        Ok(snapshot)
    }

    fn validate_payload(&self) -> Result<()> {
        if self.version != GDN_STATE_VERSION {
            candle_core::bail!("unsupported GDN state version {}", self.version)
        }
        if self.prefix_tokens == 0 {
            candle_core::bail!("GDN state requires a nonempty prefill boundary")
        }
        if self.layout.model_layer_indices.is_empty()
            || self
                .layout
                .model_layer_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.layout.tensor_parallel_world_size == 0
            || self.layout.tensor_parallel_rank >= self.layout.tensor_parallel_world_size
            || self.layout.conv_dtype != GdnStateDType::F32
            || self.layout.recurrent_dtype != GdnStateDType::F32
            || self.layout.conv_shape.contains(&0)
            || self.layout.recurrent_shape.contains(&0)
        {
            candle_core::bail!("invalid GDN layer count, dtype, or shape")
        }
        let expected = self
            .layout
            .bytes_per_layer()?
            .checked_mul(self.layout.model_layer_indices.len())
            .ok_or_else(|| candle_core::Error::Msg("GDN state byte count overflow".into()))?;
        if self.payload.len() != expected {
            candle_core::bail!(
                "GDN state payload length mismatch: got {}, expected {}",
                self.payload.len(),
                expected
            )
        }
        if state_bytes::sha256(&self.payload) != self.payload_sha256 {
            candle_core::bail!("GDN state payload SHA-256 mismatch")
        }
        Ok(())
    }
}

fn to_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("GDN state dimension exceeds u32".into()))
}

fn checked_elements(shape: &[u32]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim as usize)
            .ok_or_else(|| candle_core::Error::Msg("GDN state shape overflow".into()))
    })
}

fn tensor_from_f32_bits(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    Tensor::from_vec(state_bytes::f32_from_le_bytes(bytes)?, shape, device)
}

pub fn export_gdn_state(
    cache: &MambaCache,
    layout: &GdnStateLayout,
    seq_id: usize,
    prefix_tokens: u64,
    model_fingerprint: [u8; 32],
) -> Result<GdnStateSnapshot> {
    layout.validate_cache(cache)?;
    if prefix_tokens == 0 {
        candle_core::bail!("GDN state requires a nonempty prefill boundary")
    }
    let slot = cache
        .get_slot(seq_id)
        .ok_or_else(|| candle_core::Error::Msg(format!("GDN sequence {seq_id} has no slot")))?;
    let capacity = layout
        .bytes_per_layer()?
        .checked_mul(cache.num_gdn_layers())
        .ok_or_else(|| candle_core::Error::Msg("GDN state byte count overflow".into()))?;
    let mut payload = Vec::with_capacity(capacity);
    for layer in 0..cache.num_gdn_layers() {
        state_bytes::append_f32_bits(&cache.get_conv_state(layer, slot)?, &mut payload)?;
        state_bytes::append_f32_bits(&cache.get_recurrent_state(layer, slot)?, &mut payload)?;
    }
    let snapshot = GdnStateSnapshot {
        version: GDN_STATE_VERSION,
        prefix_tokens,
        model_fingerprint,
        layout: layout.clone(),
        payload_sha256: state_bytes::sha256(&payload),
        payload,
    };
    Ok(snapshot)
}

/// Export the existing v1 envelope directly, without rehashing a snapshot
/// just constructed from the cache.
pub fn export_gdn_state_bytes(
    cache: &MambaCache,
    layout: &GdnStateLayout,
    seq_id: usize,
    prefix_tokens: u64,
    model_fingerprint: [u8; 32],
) -> Result<Vec<u8>> {
    export_gdn_state(cache, layout, seq_id, prefix_tokens, model_fingerprint)?
        .to_bytes_after_export()
}

pub fn import_gdn_state(
    cache: &mut MambaCache,
    layout: &GdnStateLayout,
    seq_id: usize,
    expected_prefix_tokens: u64,
    expected_model_fingerprint: [u8; 32],
    snapshot: &GdnStateSnapshot,
) -> Result<()> {
    snapshot.validate_payload()?;
    import_gdn_state_validated(
        cache,
        layout,
        seq_id,
        expected_prefix_tokens,
        expected_model_fingerprint,
        snapshot,
    )
}

/// Parse and validate a v1 envelope once, then import it while the parsed
/// snapshot remains private and immutable.
pub fn import_gdn_state_bytes(
    cache: &mut MambaCache,
    layout: &GdnStateLayout,
    seq_id: usize,
    expected_prefix_tokens: u64,
    expected_model_fingerprint: [u8; 32],
    bytes: &[u8],
) -> Result<()> {
    if cache.get_slot(seq_id).is_some() {
        candle_core::bail!("GDN sequence {seq_id} already has a slot; release it before import")
    }
    let snapshot = GdnStateSnapshot::from_bytes(bytes)?;
    import_gdn_state_validated(
        cache,
        layout,
        seq_id,
        expected_prefix_tokens,
        expected_model_fingerprint,
        &snapshot,
    )
}

fn import_gdn_state_validated(
    cache: &mut MambaCache,
    layout: &GdnStateLayout,
    seq_id: usize,
    expected_prefix_tokens: u64,
    expected_model_fingerprint: [u8; 32],
    snapshot: &GdnStateSnapshot,
) -> Result<()> {
    layout.validate_cache(cache)?;
    if &snapshot.layout != layout {
        candle_core::bail!("GDN state layer order, shape, dtype, or TP layout mismatch")
    }
    if snapshot.prefix_tokens != expected_prefix_tokens {
        candle_core::bail!("GDN state prefill boundary mismatch")
    }
    if snapshot.model_fingerprint != expected_model_fingerprint {
        candle_core::bail!("GDN state model fingerprint mismatch")
    }
    if cache.get_slot(seq_id).is_some() {
        candle_core::bail!("GDN sequence {seq_id} already has a slot; release it before import")
    }

    let slot = cache.allocate_slot(seq_id)?;
    let device = cache.conv_state(0).device().clone();
    let conv_bytes = checked_elements(&layout.conv_shape)? * 4;
    let recurrent_bytes = checked_elements(&layout.recurrent_shape)? * 4;
    let conv_shape = [
        1,
        layout.conv_shape[0] as usize,
        layout.conv_shape[1] as usize,
    ];
    let recurrent_shape = [
        1,
        layout.recurrent_shape[0] as usize,
        layout.recurrent_shape[1] as usize,
        layout.recurrent_shape[2] as usize,
    ];
    let result = (|| {
        let slots = Tensor::from_vec(vec![slot as i64], (1,), &device)?;
        let mut offset = 0;
        for layer in 0..cache.num_gdn_layers() {
            let conv = tensor_from_f32_bits(
                &snapshot.payload[offset..offset + conv_bytes],
                &conv_shape,
                &device,
            )?;
            offset += conv_bytes;
            let recurrent = tensor_from_f32_bits(
                &snapshot.payload[offset..offset + recurrent_bytes],
                &recurrent_shape,
                &device,
            )?;
            offset += recurrent_bytes;
            cache.set_batch_conv_state(layer, &slots, &conv)?;
            cache.set_batch_recurrent_state(layer, &slots, &recurrent)?;
        }
        Ok(())
    })();
    if result.is_err() {
        cache.free_slot(seq_id);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> MambaCache {
        MambaCache::new(2, 2, 4, 4, 2, 2, 2, DType::F32, DType::F32, &Device::Cpu).unwrap()
    }

    fn layout(cache: &MambaCache) -> GdnStateLayout {
        GdnStateLayout::from_cache(cache, vec![0, 2], 0, 1).unwrap()
    }

    fn filled_cache() -> MambaCache {
        let mut cache = cache();
        let slot = cache.allocate_slot(7).unwrap();
        let slots = Tensor::from_vec(vec![slot as i64], (1,), &Device::Cpu).unwrap();
        for layer in 0usize..2 {
            let mut conv = (0..12)
                .map(|i| f32::from_bits(0x3f80_0000 + i + (layer * 100) as u32))
                .collect::<Vec<_>>();
            conv[0] = f32::from_bits(0x7fc0_1234); // Preserve NaN payload bits.
            conv[1] = -0.0;
            let recurrent = (0..8)
                .map(|i| f32::from_bits(0x4000_0000 + i + (layer * 100) as u32))
                .collect::<Vec<_>>();
            cache
                .set_batch_conv_state(
                    layer,
                    &slots,
                    &Tensor::from_vec(conv, (1, 4, 3), &Device::Cpu).unwrap(),
                )
                .unwrap();
            cache
                .set_batch_recurrent_state(
                    layer,
                    &slots,
                    &Tensor::from_vec(recurrent, (1, 2, 2, 2), &Device::Cpu).unwrap(),
                )
                .unwrap();
        }
        cache
    }

    #[test]
    fn bulk_bytes_keep_v1_wire_format() {
        #[derive(Serialize)]
        struct Legacy<'a> {
            version: u32,
            prefix_tokens: u64,
            model_fingerprint: &'a [u8; 32],
            layout: &'a GdnStateLayout,
            payload_sha256: &'a [u8; 32],
            payload: &'a Vec<u8>,
        }
        let source = filled_cache();
        let snapshot = export_gdn_state(&source, &layout(&source), 7, 41, [3; 32]).unwrap();
        let legacy = Legacy {
            version: snapshot.version,
            prefix_tokens: snapshot.prefix_tokens,
            model_fingerprint: &snapshot.model_fingerprint,
            layout: &snapshot.layout,
            payload_sha256: &snapshot.payload_sha256,
            payload: &snapshot.payload,
        };
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(
            &bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .serialize(&legacy)
                .unwrap(),
        );
        assert_eq!(snapshot.to_bytes().unwrap(), bytes);
        assert_eq!(
            GdnStateSnapshot::from_bytes(&bytes).unwrap().payload,
            snapshot.payload
        );
    }

    #[test]
    fn bytes_first_roundtrip_and_rejections() {
        let source = filled_cache();
        let source_layout = layout(&source);
        let wire = export_gdn_state_bytes(&source, &source_layout, 7, 41, [3; 32]).unwrap();
        let ordinary = export_gdn_state(&source, &source_layout, 7, 41, [3; 32]).unwrap();
        assert_eq!(wire, ordinary.to_bytes().unwrap());
        let mut target = cache();
        let target_layout = layout(&target);
        assert!(
            import_gdn_state_bytes(&mut target, &target_layout, 99, 42, [3; 32], &wire).is_err()
        );
        assert!(
            import_gdn_state_bytes(&mut target, &target_layout, 99, 41, [4; 32], &wire).is_err()
        );
        let mut corrupt = wire.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(
            import_gdn_state_bytes(&mut target, &target_layout, 99, 41, [3; 32], &corrupt).is_err()
        );
        assert_eq!(target.get_slot(99), None);
        import_gdn_state_bytes(&mut target, &target_layout, 99, 41, [3; 32], &wire).unwrap();
        assert!(
            import_gdn_state_bytes(&mut target, &target_layout, 99, 41, [3; 32], &wire).is_err()
        );
        let actual = export_gdn_state(&target, &target_layout, 99, 41, [3; 32]).unwrap();
        assert_eq!(actual.payload, ordinary.payload);
    }

    #[test]
    fn independent_cache_roundtrip_keeps_every_fp32_bit() {
        let source = filled_cache();
        let expected = export_gdn_state(&source, &layout(&source), 7, 41, [3; 32]).unwrap();
        let encoded = expected.to_bytes().unwrap();
        let decoded = GdnStateSnapshot::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.payload, expected.payload);

        let mut target = cache();
        target.allocate_slot(8).unwrap(); // Import must select a distinct free slot.
        let target_layout = layout(&target);
        import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &decoded).unwrap();
        assert_ne!(target.get_slot(8), target.get_slot(99));
        let actual = export_gdn_state(&target, &layout(&target), 99, 41, [3; 32]).unwrap();
        assert_eq!(actual.payload, expected.payload);
    }

    #[test]
    fn rejects_corrupt_and_incompatible_state_before_allocating_slot() {
        let source = filled_cache();
        let snapshot = export_gdn_state(&source, &layout(&source), 7, 41, [3; 32]).unwrap();
        let encoded = snapshot.to_bytes().unwrap();
        assert!(GdnStateSnapshot::from_bytes(&encoded[..encoded.len() - 1]).is_err());
        assert!(GdnStateSnapshot::from_bytes(&encoded[1..]).is_err());
        let mut target = cache();
        let target_layout = layout(&target);

        let mut bad = snapshot.clone();
        bad.version += 1;
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.payload[0] ^= 1;
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.layout.conv_shape = [5, 3];
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.layout.conv_dtype = GdnStateDType::BF16;
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.layout.model_layer_indices.pop();
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.layout.model_layer_indices = vec![0, 3];
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        let mut bad = snapshot.clone();
        bad.layout.tensor_parallel_rank = 1;
        bad.layout.tensor_parallel_world_size = 2;
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &bad).is_err());
        assert!(import_gdn_state(&mut target, &target_layout, 99, 42, [3; 32], &snapshot).is_err());
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [4; 32], &snapshot).is_err());
        assert_eq!(target.get_slot(99), None);

        import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &snapshot).unwrap();
        assert!(import_gdn_state(&mut target, &target_layout, 99, 41, [3; 32], &snapshot).is_err());
    }
}
