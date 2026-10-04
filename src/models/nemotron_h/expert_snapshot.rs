//! Portable host representation of one Nemotron-H routed expert.
//!
//! The payload contains the checkpoint's original tensor bytes, including
//! packed NVFP4 weights and FP8 block scales. A caller-supplied fingerprint
//! binds the snapshot to weights, adapters, and numerical settings; the model
//! layout digest additionally checks every checkpoint tensor's name, dtype,
//! and shape without reading all weight payloads.

use crate::models::layers::state_bytes;
use bincode::Options;
use candle_core::{DType, Result};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 8] = b"XNEX\0\0\0\x01";
pub const NEMOTRON_EXPERT_SNAPSHOT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NemotronExpertTensorDType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E8M0,
}

impl NemotronExpertTensorDType {
    pub(super) fn from_safetensors(name: &str) -> Result<Self> {
        Ok(match name {
            "U8" => Self::U8,
            "U32" => Self::U32,
            "I64" => Self::I64,
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E8M0" => Self::F8E8M0,
            _ => candle_core::bail!("unsupported Nemotron expert tensor dtype {name}"),
        })
    }

    pub(super) fn candle(self) -> DType {
        match self {
            Self::U8 => DType::U8,
            Self::U32 => DType::U32,
            Self::I64 => DType::I64,
            Self::BF16 => DType::BF16,
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
            Self::F8E4M3 => DType::F8E4M3,
            Self::F8E8M0 => DType::F8E8M0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NemotronExpertTensorSpec {
    /// Projection-relative name, such as `up_proj.weight_packed`.
    pub name: String,
    pub dtype: NemotronExpertTensorDType,
    pub shape: Vec<u32>,
    pub byte_len: u64,
}

impl NemotronExpertTensorSpec {
    fn validate(&self) -> Result<usize> {
        if !(self.name.starts_with("up_proj.") || self.name.starts_with("down_proj."))
            || self.shape.len() > 4
            || self.shape.contains(&0)
        {
            candle_core::bail!("invalid Nemotron expert tensor descriptor {}", self.name)
        }
        let elements = self.shape.iter().try_fold(1usize, |total, &dim| {
            total.checked_mul(dim as usize).ok_or_else(|| {
                candle_core::Error::Msg("Nemotron expert tensor shape overflow".into())
            })
        })?;
        let bytes = elements
            .checked_mul(self.dtype.candle().size_in_bytes())
            .ok_or_else(|| {
                candle_core::Error::Msg("Nemotron expert tensor byte overflow".into())
            })?;
        if self.byte_len != bytes as u64 {
            candle_core::bail!("Nemotron expert tensor {} byte length mismatch", self.name)
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NemotronExpertLayout {
    /// Digest of all model tensor names, dtypes, shapes, and the model layout.
    pub model_layout_sha256: [u8; 32],
    pub layer: u32,
    pub expert: u32,
    pub activation_dtype: String,
    pub quant_format: String,
    /// Sorted projection tensors in payload order.
    pub tensors: Vec<NemotronExpertTensorSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NemotronExpertSnapshot {
    pub version: u32,
    /// Caller-supplied identity for actual weights, adapters, and settings.
    pub model_fingerprint: [u8; 32],
    pub layout: NemotronExpertLayout,
    pub payload_sha256: [u8; 32],
    /// Concatenated raw tensor bytes in `layout.tensors` order.
    #[serde(with = "state_bytes")]
    pub payload: Vec<u8>,
}

impl NemotronExpertSnapshot {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        self.to_bytes_after_export()
    }

    /// Only use for a snapshot assembled from the owned checkpoint mapping.
    pub(super) fn to_bytes_after_export(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(MAGIC.len() + self.payload.len() + 512);
        bytes.extend_from_slice(MAGIC);
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut bytes, self)
            .map_err(|e| candle_core::Error::Msg(format!("serialize Nemotron expert: {e}")))?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let encoded = bytes.strip_prefix(MAGIC).ok_or_else(|| {
            candle_core::Error::Msg("invalid Nemotron expert snapshot magic/version".into())
        })?;
        let snapshot: Self = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(encoded.len() as u64)
            .reject_trailing_bytes()
            .deserialize(encoded)
            .map_err(|e| candle_core::Error::Msg(format!("deserialize Nemotron expert: {e}")))?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != NEMOTRON_EXPERT_SNAPSHOT_VERSION {
            candle_core::bail!(
                "unsupported Nemotron expert snapshot version {}",
                self.version
            )
        }
        if self.layout.tensors.is_empty()
            || self.layout.tensors.len() > 16
            || self.layout.activation_dtype.is_empty()
            || self.layout.quant_format.is_empty()
        {
            candle_core::bail!("invalid Nemotron expert snapshot layout")
        }
        let mut expected = 0usize;
        let mut previous = None::<&str>;
        for tensor in &self.layout.tensors {
            if previous.is_some_and(|name| name >= tensor.name.as_str()) {
                candle_core::bail!("Nemotron expert tensor names must be unique and sorted")
            }
            previous = Some(&tensor.name);
            expected = expected.checked_add(tensor.validate()?).ok_or_else(|| {
                candle_core::Error::Msg("Nemotron expert payload byte overflow".into())
            })?;
        }
        if self.payload.len() != expected {
            candle_core::bail!("Nemotron expert payload length mismatch")
        }
        if state_bytes::sha256(&self.payload) != self.payload_sha256 {
            candle_core::bail!("Nemotron expert payload SHA-256 mismatch")
        }
        Ok(())
    }

    pub(super) fn tensor(&self, name: &str) -> Result<(&NemotronExpertTensorSpec, &[u8])> {
        let mut offset = 0usize;
        for spec in &self.layout.tensors {
            let end = offset + spec.byte_len as usize;
            if spec.name == name {
                return Ok((spec, &self.payload[offset..end]));
            }
            offset = end;
        }
        candle_core::bail!("missing Nemotron expert snapshot tensor {name}")
    }

    pub(super) fn has_tensor(&self, name: &str) -> bool {
        self.layout.tensors.iter().any(|spec| spec.name == name)
    }
}
