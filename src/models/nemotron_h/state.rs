//! Portable Nemotron-H Mamba state at a completed token boundary.
//!
//! Attention KV must be transferred separately at the same boundary. The
//! caller supplies a SHA-256 identity covering weights, adapters, and
//! execution settings; loaded `VarBuilderX` weights have no intrinsic ID.

use super::{Block, MambaState, Mixer};
use crate::models::layers::state_bytes;
use bincode::Options;
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"XNMH\0\0\0\x01";
pub const NEMOTRON_MAMBA_STATE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NemotronStateDType {
    F32,
    F16,
    BF16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NemotronMambaStateLayout {
    /// Number of decoder blocks, including attention and MLP blocks.
    pub decoder_layers: u32,
    /// Absolute decoder layer indices in payload order.
    pub model_layer_indices: Vec<u32>,
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_world_size: u32,
    pub conv_dtype: NemotronStateDType,
    pub ssm_dtype: NemotronStateDType,
    /// Per-sequence convolution history `[kernel - 1, conv_dim]`, oldest first.
    pub conv_shape: [u32; 2],
    /// Per-sequence SSM state `[heads, head_dim, state_dim]`.
    pub ssm_shape: [u32; 3],
}

impl NemotronMambaStateLayout {
    fn validate(&self) -> Result<()> {
        if self.decoder_layers == 0
            || self.model_layer_indices.is_empty()
            || self
                .model_layer_indices
                .iter()
                .any(|&i| i >= self.decoder_layers)
            || self
                .model_layer_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.tensor_parallel_rank != 0
            || self.tensor_parallel_world_size != 1
            || self.conv_dtype != NemotronStateDType::F32
            || self.ssm_dtype != NemotronStateDType::F32
            || self.conv_shape.contains(&0)
            || self.ssm_shape.contains(&0)
        {
            candle_core::bail!("invalid Nemotron-H Mamba state layout")
        }
        Ok(())
    }

    fn bytes_per_layer(&self) -> Result<usize> {
        checked_elements(&self.conv_shape)?
            .checked_add(checked_elements(&self.ssm_shape)?)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| candle_core::Error::Msg("Nemotron-H state byte count overflow".into()))
    }

    pub fn payload_bytes(&self) -> Result<usize> {
        self.validate()?;
        self.bytes_per_layer()?
            .checked_mul(self.model_layer_indices.len())
            .ok_or_else(|| candle_core::Error::Msg("Nemotron-H state byte count overflow".into()))
    }
}

pub(super) fn layout_from_layers(layers: &[Block]) -> Result<NemotronMambaStateLayout> {
    let mut indices = Vec::new();
    let mut shapes = None;
    for (i, layer) in layers.iter().enumerate() {
        if let Mixer::Mamba(mamba) = &layer.mixer {
            let conv = [to_u32(mamba.kernel - 1)?, to_u32(mamba.conv_dim)?];
            let ssm = [
                to_u32(mamba.heads)?,
                to_u32(mamba.head_dim)?,
                to_u32(mamba.state_dim)?,
            ];
            if let Some((prior_conv, prior_ssm)) = shapes {
                if (conv, ssm) != (prior_conv, prior_ssm) {
                    candle_core::bail!("Nemotron-H Mamba layers have incompatible state shapes")
                }
            }
            shapes = Some((conv, ssm));
            indices.push(to_u32(i)?);
        }
    }
    let (conv_shape, ssm_shape) = shapes
        .ok_or_else(|| candle_core::Error::Msg("Nemotron-H model has no Mamba layers".into()))?;
    let layout = NemotronMambaStateLayout {
        decoder_layers: to_u32(layers.len())?,
        model_layer_indices: indices,
        tensor_parallel_rank: 0,
        tensor_parallel_world_size: 1,
        conv_dtype: NemotronStateDType::F32,
        ssm_dtype: NemotronStateDType::F32,
        conv_shape,
        ssm_shape,
    };
    layout.validate()?;
    Ok(layout)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NemotronMambaSnapshot {
    pub version: u32,
    pub prefix_tokens: u64,
    /// Caller-supplied SHA-256 identity for weights and execution settings.
    pub model_fingerprint: [u8; 32],
    pub layout: NemotronMambaStateLayout,
    /// SHA-256 of `payload`.
    pub payload_sha256: [u8; 32],
    /// Little-endian FP32 bits, conv then SSM for each Mamba layer.
    #[serde(with = "state_bytes")]
    pub payload: Vec<u8>,
}

impl NemotronMambaSnapshot {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate_payload()?;
        self.to_bytes_after_capture()
    }

    /// Only call on a snapshot constructed by `capture`, before exposing its
    /// public, mutable fields to a caller.
    pub(super) fn to_bytes_after_capture(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(
            MAGIC.len() + self.payload.len() + self.layout.model_layer_indices.len() * 4 + 256,
        );
        bytes.extend_from_slice(MAGIC);
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut bytes, self)
            .map_err(|e| candle_core::Error::Msg(format!("serialize Nemotron-H state: {e}")))?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let encoded = bytes.strip_prefix(MAGIC).ok_or_else(|| {
            candle_core::Error::Msg("invalid Nemotron-H state magic/version".into())
        })?;
        let snapshot: Self = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(encoded.len() as u64)
            .reject_trailing_bytes()
            .deserialize(encoded)
            .map_err(|e| candle_core::Error::Msg(format!("deserialize Nemotron-H state: {e}")))?;
        snapshot.validate_payload()?;
        Ok(snapshot)
    }

    fn validate_payload(&self) -> Result<()> {
        if self.version != NEMOTRON_MAMBA_STATE_VERSION {
            candle_core::bail!("unsupported Nemotron-H state version {}", self.version)
        }
        if self.prefix_tokens == 0 {
            candle_core::bail!("Nemotron-H state requires a nonempty prefill boundary")
        }
        let expected = self.layout.payload_bytes()?;
        if self.payload.len() != expected {
            candle_core::bail!(
                "Nemotron-H state payload length mismatch: got {}, expected {}",
                self.payload.len(),
                expected
            )
        }
        if Sha256::digest(&self.payload).as_slice() != self.payload_sha256 {
            candle_core::bail!("Nemotron-H state payload SHA-256 mismatch")
        }
        Ok(())
    }

    pub(super) fn restore(
        &self,
        expected_layout: &NemotronMambaStateLayout,
        prefix_tokens: u64,
        model_fingerprint: [u8; 32],
        device: &Device,
    ) -> Result<Vec<Option<MambaState>>> {
        self.validate_payload()?;
        self.restore_validated(expected_layout, prefix_tokens, model_fingerprint, device)
    }

    /// `from_bytes` already checked the payload; this snapshot must remain
    /// private to the bytes-first import call until installation completes.
    pub(super) fn restore_validated(
        &self,
        expected_layout: &NemotronMambaStateLayout,
        prefix_tokens: u64,
        model_fingerprint: [u8; 32],
        device: &Device,
    ) -> Result<Vec<Option<MambaState>>> {
        if &self.layout != expected_layout {
            candle_core::bail!("Nemotron-H state layout does not match target model")
        }
        if self.prefix_tokens != prefix_tokens {
            candle_core::bail!("Nemotron-H state prefix boundary mismatch")
        }
        if self.model_fingerprint != model_fingerprint {
            candle_core::bail!("Nemotron-H state model fingerprint mismatch")
        }
        let conv_shape = self.layout.conv_shape.map(|v| v as usize);
        let ssm_shape = self.layout.ssm_shape.map(|v| v as usize);
        let conv_count = checked_elements(&self.layout.conv_shape)?;
        let ssm_count = checked_elements(&self.layout.ssm_shape)?;
        let conv_bytes = conv_count * 4;
        let ssm_bytes = ssm_count * 4;
        let mut offset = 0;
        let mut states = vec![None; self.layout.decoder_layers as usize];
        for &layer in &self.layout.model_layer_indices {
            let conv = state_bytes::f32_from_le_bytes(&self.payload[offset..offset + conv_bytes])?;
            offset += conv_bytes;
            let ssm = state_bytes::f32_from_le_bytes(&self.payload[offset..offset + ssm_bytes])?;
            offset += ssm_bytes;
            states[layer as usize] = Some(MambaState {
                conv: Tensor::from_vec(conv, &conv_shape, device)?,
                ssm: Tensor::from_vec(ssm, &ssm_shape, device)?,
            });
        }
        debug_assert_eq!(offset, self.payload.len());
        Ok(states)
    }
}

pub(super) fn capture(
    states: &[Option<MambaState>],
    layout: NemotronMambaStateLayout,
    prefix_tokens: u64,
    model_fingerprint: [u8; 32],
) -> Result<NemotronMambaSnapshot> {
    if prefix_tokens == 0 || states.len() != layout.decoder_layers as usize {
        candle_core::bail!("Nemotron-H state has no completed prefix or wrong layer count")
    }
    let mut payload = Vec::with_capacity(layout.payload_bytes()?);
    for &layer in &layout.model_layer_indices {
        let state = states[layer as usize].as_ref().ok_or_else(|| {
            candle_core::Error::Msg(format!("Nemotron-H Mamba state missing at layer {layer}"))
        })?;
        if state.conv.dtype() != DType::F32
            || state.ssm.dtype() != DType::F32
            || state.conv.dims() != layout.conv_shape.map(|v| v as usize)
            || state.ssm.dims() != layout.ssm_shape.map(|v| v as usize)
        {
            candle_core::bail!("Nemotron-H Mamba state shape or dtype mismatch at layer {layer}")
        }
        state_bytes::append_f32_bits(&state.conv, &mut payload)?;
        state_bytes::append_f32_bits(&state.ssm, &mut payload)?;
    }
    let snapshot = NemotronMambaSnapshot {
        version: NEMOTRON_MAMBA_STATE_VERSION,
        prefix_tokens,
        model_fingerprint,
        layout,
        payload_sha256: Sha256::digest(&payload).into(),
        payload,
    };
    Ok(snapshot)
}

pub(super) fn install(
    states: &mut HashMap<usize, Vec<Option<MambaState>>>,
    seq_id: usize,
    restored: Vec<Option<MambaState>>,
    capacity: usize,
) -> Result<()> {
    if states.contains_key(&seq_id) {
        candle_core::bail!("Nemotron-H sequence {seq_id} already has Mamba state")
    }
    if states.len() >= capacity {
        candle_core::bail!("Nemotron-H recurrent state capacity exceeded")
    }
    states.insert(seq_id, restored);
    Ok(())
}

fn to_u32(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg("Nemotron-H state dimension exceeds u32".into()))
}

fn checked_elements(shape: &[u32]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim as usize)
            .ok_or_else(|| candle_core::Error::Msg("Nemotron-H state shape overflow".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> NemotronMambaStateLayout {
        NemotronMambaStateLayout {
            decoder_layers: 4,
            model_layer_indices: vec![0, 2],
            tensor_parallel_rank: 0,
            tensor_parallel_world_size: 1,
            conv_dtype: NemotronStateDType::F32,
            ssm_dtype: NemotronStateDType::F32,
            conv_shape: [2, 3],
            ssm_shape: [2, 2, 2],
        }
    }

    fn fixture() -> Result<(NemotronMambaStateLayout, NemotronMambaSnapshot)> {
        let layout = layout();
        let mut states = vec![None; 4];
        let conv = Tensor::from_vec(vec![-0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3), &Device::Cpu)?;
        let ssm = Tensor::from_vec((0..8).map(|v| v as f32).collect(), (2, 2, 2), &Device::Cpu)?;
        states[0] = Some(MambaState {
            conv: conv.clone(),
            ssm: ssm.clone(),
        });
        states[2] = Some(MambaState { conv, ssm });
        let snapshot = capture(&states, layout.clone(), 12, [7; 32])?;
        Ok((layout, snapshot))
    }

    #[test]
    fn bulk_bytes_keep_v1_wire_format() -> Result<()> {
        #[derive(Serialize)]
        struct Legacy<'a> {
            version: u32,
            prefix_tokens: u64,
            model_fingerprint: &'a [u8; 32],
            layout: &'a NemotronMambaStateLayout,
            payload_sha256: &'a [u8; 32],
            payload: &'a Vec<u8>,
        }
        let (_, snapshot) = fixture()?;
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
                .map_err(candle_core::Error::wrap)?,
        );
        assert_eq!(snapshot.to_bytes()?, bytes);
        assert_eq!(
            NemotronMambaSnapshot::from_bytes(&bytes)?.payload,
            snapshot.payload
        );
        Ok(())
    }

    #[test]
    fn roundtrip_into_independent_sequence_preserves_fp32_bits() -> Result<()> {
        let (layout, snapshot) = fixture()?;
        let bytes = snapshot.to_bytes()?;
        let decoded = NemotronMambaSnapshot::from_bytes(&bytes)?;
        let restored = decoded.restore(&layout, 12, [7; 32], &Device::Cpu)?;
        let mut target = HashMap::new();
        install(&mut target, 99, restored, 1)?;
        let second = capture(target.get(&99).unwrap(), layout, 12, [7; 32])?;
        assert_eq!(snapshot.payload, second.payload);
        assert_eq!(&snapshot.payload[..4], &(-0.0f32).to_bits().to_le_bytes());
        assert!(install(&mut target, 99, vec![None; 4], 2).is_err());
        assert!(install(&mut target, 100, vec![None; 4], 1).is_err());
        assert_eq!(target.len(), 1);
        Ok(())
    }

    #[test]
    fn rejects_corruption_and_incompatible_imports() -> Result<()> {
        let (layout, snapshot) = fixture()?;
        let mut bytes = snapshot.to_bytes()?;
        bytes[0] ^= 1;
        assert!(NemotronMambaSnapshot::from_bytes(&bytes).is_err());
        let mut bytes = snapshot.to_bytes()?;
        bytes.pop();
        assert!(NemotronMambaSnapshot::from_bytes(&bytes).is_err());
        let mut corrupt = snapshot.clone();
        corrupt.payload[0] ^= 1;
        assert!(corrupt.to_bytes().is_err());
        let mut wrong = snapshot.clone();
        wrong.version += 1;
        assert!(wrong.to_bytes().is_err());
        let mut wrong = snapshot.clone();
        wrong.layout.conv_shape[0] += 1;
        assert!(wrong.to_bytes().is_err());
        let mut wrong = snapshot.clone();
        wrong.layout.model_layer_indices.swap(0, 1);
        assert!(wrong.to_bytes().is_err());
        let mut wrong = snapshot.clone();
        wrong.layout.conv_dtype = NemotronStateDType::BF16;
        assert!(wrong.to_bytes().is_err());
        assert!(snapshot
            .restore(&layout, 13, [7; 32], &Device::Cpu)
            .is_err());
        assert!(snapshot
            .restore(&layout, 12, [8; 32], &Device::Cpu)
            .is_err());
        let mut other_layout = layout;
        other_layout.model_layer_indices = vec![1, 2];
        assert!(snapshot
            .restore(&other_layout, 12, [7; 32], &Device::Cpu)
            .is_err());
        Ok(())
    }
}
