//! Bulk byte encoding for portable recurrent-state snapshots.
//!
//! With bincode's fixed-int encoding, both `Vec<u8>` and `serialize_bytes`
//! use a u64 length followed by the same bytes. The latter makes one bulk
//! write instead of serializing each byte as a sequence element.

use serde::de::{self, Deserializer, Visitor};
use serde::Serializer;
use std::fmt;

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    let digest = ring::digest::digest(&ring::digest::SHA256, bytes);
    let mut result = [0u8; 32];
    result.copy_from_slice(digest.as_ref());
    result
}

pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(bytes)
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    struct ByteVisitor;

    impl<'de> Visitor<'de> for ByteVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a byte buffer")
        }

        fn visit_byte_buf<E: de::Error>(self, bytes: Vec<u8>) -> Result<Self::Value, E> {
            Ok(bytes)
        }

        fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
            Ok(bytes.to_vec())
        }
    }

    deserializer.deserialize_byte_buf(ByteVisitor)
}

pub(crate) fn append_f32_bits(
    tensor: &candle_core::Tensor,
    payload: &mut Vec<u8>,
) -> candle_core::Result<()> {
    let values = tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    #[cfg(target_endian = "little")]
    payload.extend_from_slice(bytemuck::cast_slice(&values));
    #[cfg(not(target_endian = "little"))]
    for value in values {
        payload.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(())
}

pub(crate) fn f32_from_le_bytes(bytes: &[u8]) -> candle_core::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        candle_core::bail!("recurrent-state FP32 bytes are not word aligned")
    }
    #[cfg(target_endian = "little")]
    {
        // `f32` accepts every bit pattern, including NaNs with payload bits.
        let mut values = vec![0.0f32; bytes.len() / 4];
        bytemuck::cast_slice_mut::<f32, u8>(&mut values).copy_from_slice(bytes);
        Ok(values)
    }
    #[cfg(not(target_endian = "little"))]
    {
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_float_bits_and_rejects_partial_word() {
        let bits = [0x8000_0000u32, 0x7fc0_1234, 0x3f80_0000];
        let bytes = bits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let values = f32_from_le_bytes(&bytes).unwrap();
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        assert!(f32_from_le_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn accelerated_sha256_matches_existing_digest() {
        use sha2::Digest;
        let bytes = (0..8193).map(|v| (v % 251) as u8).collect::<Vec<_>>();
        assert_eq!(
            sha256(&bytes).as_slice(),
            sha2::Sha256::digest(&bytes).as_slice()
        );
    }
}
