//! Validated import of shifou's `SHMS` v1 stored Mamba groups. The format is
//! intentionally parsed here without depending on shifou's storage layer.
use super::state::{NemotronMambaSnapshotHeader, NemotronMambaStateLayout};
use super::MambaState;
use crate::models::layers::state_bytes;
use candle_core::{Device, Result, Tensor};
use half::f16;

#[cfg(feature = "cuda")]
mod cuda;

const MAGIC: &[u8; 8] = b"SHMS\0\0\0\x01";
const MAX_NATIVE_BYTES: usize = 1024 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 1024 * 1024;
const MAX_GROUPS: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct GpuGroup {
    pub code_offset: u32,
    pub mode: u32,
    /// INT8 scale or scaled-FP16 factor. Exact groups use 1.
    pub factor: f32,
}

#[cfg(feature = "cuda")]
unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for GpuGroup {}

pub(super) struct CompactMambaFrame<'a> {
    data: &'a [u8],
    groups: Vec<GpuGroup>,
    conv_offset: usize,
    layers: usize,
    conv_words: usize,
    heads: usize,
    values: usize,
    channels: usize,
    layout: &'a NemotronMambaStateLayout,
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(count)
        .ok_or_else(|| candle_core::Error::Msg("compact Mamba state offset overflow".into()))?;
    let part = bytes
        .get(*cursor..end)
        .ok_or_else(|| candle_core::Error::Msg("truncated compact Mamba state".into()))?;
    *cursor = end;
    Ok(part)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<usize> {
    Ok(u32::from_le_bytes(take(bytes, cursor, 4)?.try_into().unwrap()) as usize)
}

fn product(dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| candle_core::Error::Msg("compact Mamba state dimension overflow".into()))
    })
}

impl<'a> CompactMambaFrame<'a> {
    pub(super) fn parse(
        bytes: &'a [u8],
        layout: &'a NemotronMambaStateLayout,
        prefix_tokens: u64,
        model_fingerprint: [u8; 32],
    ) -> Result<Self> {
        if bytes.len() < 32 || bytes.len() > MAX_NATIVE_BYTES {
            candle_core::bail!("compact Mamba state length exceeds bounds")
        }
        let (data, digest) = bytes.split_at(bytes.len() - 32);
        if state_bytes::sha256(data).as_slice() != digest {
            candle_core::bail!("compact Mamba state SHA-256 mismatch")
        }
        let mut cursor = 0;
        if take(data, &mut cursor, 8)? != MAGIC {
            candle_core::bail!("unsupported compact Mamba state version")
        }
        let header_len = read_u32(data, &mut cursor)?;
        let layers = read_u32(data, &mut cursor)?;
        let conv_words = read_u32(data, &mut cursor)?;
        let heads = read_u32(data, &mut cursor)?;
        let values = read_u32(data, &mut cursor)?;
        let channels = read_u32(data, &mut cursor)?;
        let group_count = read_u32(data, &mut cursor)?;
        if !(40..=MAX_HEADER_BYTES).contains(&header_len) || group_count > MAX_GROUPS {
            candle_core::bail!("compact Mamba state header or group count exceeds limit")
        }
        let expected_layers = layout.model_layer_indices.len();
        let expected_conv = product(&layout.conv_shape.map(|v| v as usize))?;
        let [expected_heads, expected_values, expected_channels] =
            layout.ssm_shape.map(|v| v as usize);
        let expected_groups = product(&[expected_layers, expected_heads, expected_channels])?;
        if (layers, conv_words, heads, values, channels, group_count)
            != (
                expected_layers,
                expected_conv,
                expected_heads,
                expected_values,
                expected_channels,
                expected_groups,
            )
        {
            candle_core::bail!("compact Mamba state shape does not match target model")
        }
        let native_payload = layout.payload_bytes()?;
        if native_payload > MAX_NATIVE_BYTES || header_len > MAX_NATIVE_BYTES - native_payload {
            candle_core::bail!("compact Mamba state native size exceeds limit")
        }
        let header = NemotronMambaSnapshotHeader::from_bytes(take(data, &mut cursor, header_len)?)?;
        if &header.layout != layout {
            candle_core::bail!("compact Mamba state layout does not match target model")
        }
        if header.prefix_tokens != prefix_tokens {
            candle_core::bail!("compact Mamba state prefix boundary mismatch")
        }
        if header.model_fingerprint != model_fingerprint {
            candle_core::bail!("compact Mamba state model fingerprint mismatch")
        }
        let conv_offset = cursor;
        let conv_bytes = product(&[layers, conv_words, 4])?;
        take(data, &mut cursor, conv_bytes)?;
        let modes = take(data, &mut cursor, group_count)?;
        let mut groups = Vec::with_capacity(group_count);
        for (index, &mode) in modes.iter().enumerate() {
            if mode == 0 && layout.model_layer_indices[index / (heads * channels)] == 0 {
                candle_core::bail!("Nemotron-H layer 0 cannot use INT8 recurrent groups")
            }
            let (factor, width) = match mode {
                0 => {
                    let scale = f32::from_le_bytes(take(data, &mut cursor, 4)?.try_into().unwrap());
                    if !scale.is_finite() || scale <= 0.0 {
                        candle_core::bail!("invalid compact Mamba INT8 scale")
                    }
                    (scale, 1)
                }
                1 => {
                    let exponent = take(data, &mut cursor, 1)?[0] as i8;
                    if !(-64..=64).contains(&exponent) {
                        candle_core::bail!("invalid compact Mamba FP16 exponent")
                    }
                    (2f32.powi(exponent as i32), 2)
                }
                2 => (1.0, 4),
                _ => candle_core::bail!("unknown compact Mamba group mode"),
            };
            let code_offset = u32::try_from(cursor).map_err(|_| {
                candle_core::Error::Msg("compact Mamba code offset exceeds u32".into())
            })?;
            take(
                data,
                &mut cursor,
                values.checked_mul(width).ok_or_else(|| {
                    candle_core::Error::Msg("compact Mamba group width overflow".into())
                })?,
            )?;
            groups.push(GpuGroup {
                code_offset,
                mode: mode as u32,
                factor,
            });
        }
        if cursor != data.len() {
            candle_core::bail!("trailing compact Mamba state bytes")
        }
        Ok(Self {
            data,
            groups,
            conv_offset,
            layers,
            conv_words,
            heads,
            values,
            channels,
            layout,
        })
    }

    pub(super) fn expand(&self, device: &Device) -> Result<Vec<Option<MambaState>>> {
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            return cuda::expand(self, device);
        }
        if !matches!(device, Device::Cpu) {
            candle_core::bail!("compact Mamba state expansion requires CPU or CUDA")
        }
        let conv_shape = self.layout.conv_shape.map(|v| v as usize);
        let ssm_shape = self.layout.ssm_shape.map(|v| v as usize);
        let mut states = vec![None; self.layout.decoder_layers as usize];
        let conv_bytes = self.conv_words * 4;
        for layer in 0..self.layers {
            let start = self.conv_offset + layer * conv_bytes;
            let conv = state_bytes::f32_from_le_bytes(&self.data[start..start + conv_bytes])?;
            let mut ssm = vec![0f32; self.heads * self.values * self.channels];
            for head in 0..self.heads {
                for channel in 0..self.channels {
                    let group = self.groups
                        [layer * self.heads * self.channels + head * self.channels + channel];
                    let offset = group.code_offset as usize;
                    for value in 0..self.values {
                        let result = match group.mode {
                            0 => (self.data[offset + value] as i8 as f32) * group.factor,
                            1 => {
                                f16::from_bits(u16::from_le_bytes(
                                    self.data[offset + value * 2..offset + value * 2 + 2]
                                        .try_into()
                                        .unwrap(),
                                ))
                                .to_f32()
                                    / group.factor
                            }
                            _ => f32::from_le_bytes(
                                self.data[offset + value * 4..offset + value * 4 + 4]
                                    .try_into()
                                    .unwrap(),
                            ),
                        };
                        ssm[(head * self.values + value) * self.channels + channel] = result;
                    }
                }
            }
            states[self.layout.model_layer_indices[layer] as usize] = Some(MambaState {
                conv: Tensor::from_vec(conv, &conv_shape, device)?,
                ssm: Tensor::from_vec(ssm, &ssm_shape, device)?,
            });
        }
        Ok(states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nemotron_h::state::{capture, install, NemotronStateDType};
    use std::collections::HashMap;

    fn fixture() -> Result<(NemotronMambaStateLayout, Vec<u8>, Vec<u8>)> {
        let layout = NemotronMambaStateLayout {
            decoder_layers: 3,
            model_layer_indices: vec![0, 2],
            tensor_parallel_rank: 0,
            tensor_parallel_world_size: 1,
            conv_dtype: NemotronStateDType::F32,
            ssm_dtype: NemotronStateDType::F32,
            conv_shape: [2, 3],
            ssm_shape: [2, 2, 2],
        };
        let mut states = vec![None; 3];
        for layer in [0, 2] {
            states[layer] = Some(MambaState {
                conv: Tensor::from_vec(
                    vec![-0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
                    (2, 3),
                    &Device::Cpu,
                )?,
                ssm: Tensor::from_vec((0..8).map(|x| x as f32).collect(), (2, 2, 2), &Device::Cpu)?,
            });
        }
        let snapshot = capture(&states, layout.clone(), 12, [7; 32])?;
        let native = snapshot.to_bytes()?;
        let header_len = native.len() - snapshot.payload.len();
        let mut packed = MAGIC.to_vec();
        for field in [header_len, 2, 6, 2, 2, 2, 8] {
            packed.extend_from_slice(&(field as u32).to_le_bytes());
        }
        packed.extend_from_slice(&native[..header_len]);
        for layer in 0..2 {
            let offset = layer * (6 + 8) * 4;
            packed.extend_from_slice(&snapshot.payload[offset..offset + 24]);
        }
        let modes = [1u8, 2, 1, 2, 0, 1, 2, 0];
        packed.extend_from_slice(&modes);
        for (group, &mode) in modes.iter().enumerate() {
            let head = group / 2 % 2;
            let channel = group % 2;
            let layer = group / 4;
            let layer_start = layer * (6 + 8) * 4 + 24;
            if mode == 0 {
                packed.extend_from_slice(&1f32.to_le_bytes());
            } else if mode == 1 {
                packed.push(1); // exponent +1, exact for these test values
            }
            for value in 0..2 {
                let word = layer_start + ((head * 2 + value) * 2 + channel) * 4;
                let x = f32::from_le_bytes(snapshot.payload[word..word + 4].try_into().unwrap());
                match mode {
                    0 => packed.push(x as i8 as u8),
                    1 => packed.extend_from_slice(&f16::from_f32(x * 2.0).to_bits().to_le_bytes()),
                    _ => packed.extend_from_slice(&x.to_le_bytes()),
                }
            }
        }
        packed.extend_from_slice(&state_bytes::sha256(&packed));
        Ok((layout, native, packed))
    }

    fn rehash(bytes: &mut [u8]) {
        let end = bytes.len() - 32;
        let digest = state_bytes::sha256(&bytes[..end]);
        bytes[end..].copy_from_slice(&digest);
    }

    #[test]
    fn all_modes_match_native_bits_and_cross_instance_install() -> Result<()> {
        let (layout, native, packed) = fixture()?;
        let frame = CompactMambaFrame::parse(&packed, &layout, 12, [7; 32])?;
        let restored = frame.expand(&Device::Cpu)?;
        let mut target = HashMap::new();
        install(&mut target, 99, restored, 1)?;
        let rebuilt = capture(target.get(&99).unwrap(), layout.clone(), 12, [7; 32])?;
        let original = crate::models::nemotron_h::NemotronMambaSnapshot::from_bytes(&native)?;
        assert_eq!(rebuilt.payload, original.payload);
        assert!(install(&mut target, 99, vec![None; 3], 2).is_err());
        #[cfg(feature = "cuda")]
        if let Ok(device) = Device::new_cuda(0) {
            let cuda_states = frame.expand(&device)?;
            let cuda_snapshot = capture(&cuda_states, layout, 12, [7; 32])?;
            assert_eq!(cuda_snapshot.payload, original.payload);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_matches_cpu_oracle_for_half_subnormals_and_scales() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(_) => return Ok(()),
        };
        let (layout, _, mut packed) = fixture()?;
        let offsets = {
            let frame = CompactMambaFrame::parse(&packed, &layout, 12, [7; 32])?;
            [
                frame.groups[0].code_offset as usize,
                frame.groups[4].code_offset as usize,
            ]
        };
        // Scaled-FP16 subnormal and signed zero; non-unit INT8 scale and a
        // negative code. These exercise byte-level parity with the CPU oracle.
        packed[offsets[0]..offsets[0] + 4].copy_from_slice(&[1, 0, 0, 128]);
        packed[offsets[1] - 4..offsets[1]].copy_from_slice(&0.25f32.to_le_bytes());
        packed[offsets[1]..offsets[1] + 2].copy_from_slice(&[(-3i8) as u8, 127]);
        rehash(&mut packed);
        let frame = CompactMambaFrame::parse(&packed, &layout, 12, [7; 32])?;
        let cpu = capture(&frame.expand(&Device::Cpu)?, layout.clone(), 12, [7; 32])?;
        let gpu = capture(&frame.expand(&device)?, layout, 12, [7; 32])?;
        assert_eq!(cpu.payload, gpu.payload);
        Ok(())
    }

    #[test]
    fn rejects_corruption_and_incompatible_groups() -> Result<()> {
        let (layout, _, packed) = fixture()?;
        let parse =
            |bytes: &[u8]| CompactMambaFrame::parse(bytes, &layout, 12, [7; 32]).map(|_| ());
        let mut bad = packed.clone();
        bad[40] ^= 1;
        assert!(parse(&bad).is_err()); // digest
        assert!(CompactMambaFrame::parse(&packed, &layout, 13, [7; 32]).is_err());
        assert!(CompactMambaFrame::parse(&packed, &layout, 12, [8; 32]).is_err());
        let mut wrong_layout = layout.clone();
        wrong_layout.model_layer_indices = vec![0, 1];
        assert!(CompactMambaFrame::parse(&packed, &wrong_layout, 12, [7; 32]).is_err());
        let mut bad = packed.clone();
        bad[20..24].copy_from_slice(&3u32.to_le_bytes()); // heads
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let header_len = u32::from_le_bytes(packed[8..12].try_into().unwrap()) as usize;
        let modes_start = 36 + header_len + 2 * 6 * 4;
        let mut bad = packed.clone();
        bad[modes_start] = 0; // layer 0 INT8 forbidden
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let mut bad = packed.clone();
        bad[modes_start] = 3;
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let mut bad = packed.clone();
        bad[36 + 8..36 + 12].copy_from_slice(&2u32.to_le_bytes()); // embedded native version
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let mut bad = packed.clone();
        bad[36 + header_len - 8..36 + header_len].copy_from_slice(&1u64.to_le_bytes());
        rehash(&mut bad);
        assert!(parse(&bad).is_err()); // embedded native payload length
        let mut bad = packed.clone();
        bad[modes_start + 8] = 127; // first group's scaled-FP16 exponent
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let mut bad = packed.clone();
        bad[modes_start + 8 + 5 + 8 + 5 + 8..][..4].fill(0); // INT8 scale
        rehash(&mut bad);
        assert!(parse(&bad).is_err());
        let mut bad = packed.clone();
        bad.pop();
        assert!(parse(&bad).is_err());
        Ok(())
    }

    /// A reproducible import-phase comparison at the Japanese 9B state shape.
    /// It deliberately excludes peer transport, attention KV, and model load.
    #[test]
    #[ignore]
    fn benchmark_9b_shape_import() -> Result<()> {
        use crate::models::nemotron_h::NemotronMambaSnapshot;
        use std::time::Instant;
        let device = Device::new_cuda(0)?;
        let layout = NemotronMambaStateLayout {
            decoder_layers: 56,
            model_layer_indices: vec![
                0, 2, 4, 6, 7, 9, 11, 13, 16, 18, 20, 23, 25, 27, 29, 32, 34, 36, 38, 41, 43, 44,
                46, 48, 50, 52, 54,
            ],
            tensor_parallel_rank: 0,
            tensor_parallel_world_size: 1,
            conv_dtype: NemotronStateDType::F32,
            ssm_dtype: NemotronStateDType::F32,
            conv_shape: [3, 12288],
            ssm_shape: [128, 80, 128],
        };
        let payload = vec![0u8; layout.payload_bytes()?];
        let snapshot = NemotronMambaSnapshot {
            version: 1,
            prefix_tokens: 1024,
            model_fingerprint: [7; 32],
            layout: layout.clone(),
            payload_sha256: state_bytes::sha256(&payload),
            payload,
        };
        let native = snapshot.to_bytes_after_capture()?;
        let header_len = native.len() - snapshot.payload.len();
        let groups = 27 * 128 * 128;
        let mut packed = Vec::with_capacity(55_000_000);
        packed.extend_from_slice(MAGIC);
        for field in [header_len, 27, 36864, 128, 80, 128, groups] {
            packed.extend_from_slice(&(field as u32).to_le_bytes());
        }
        packed.extend_from_slice(&native[..header_len]);
        packed.resize(packed.len() + 27 * 36864 * 4, 0);
        let mut modes = vec![0u8; groups];
        modes[..128 * 128].fill(1);
        for mode in modes.iter_mut().skip(128 * 128).step_by(5) {
            *mode = 1;
        }
        for mode in modes.iter_mut().skip(128 * 128).step_by(100) {
            *mode = 2;
        }
        packed.extend_from_slice(&modes);
        for mode in modes {
            match mode {
                0 => {
                    packed.extend_from_slice(&1f32.to_le_bytes());
                    packed.resize(packed.len() + 80, 0);
                }
                1 => {
                    packed.push(0);
                    packed.resize(packed.len() + 160, 0);
                }
                _ => packed.resize(packed.len() + 320, 0),
            }
        }
        packed.extend_from_slice(&state_bytes::sha256(&packed));
        println!(
            "native_bytes={} compact_bytes={}",
            native.len(),
            packed.len()
        );
        let frame = CompactMambaFrame::parse(&packed, &layout, 1024, [7; 32])?;
        drop(frame.expand(&device)?); // compile and warm the CUDA path
        drop(snapshot.restore_validated(&layout, 1024, [7; 32], &device)?);
        for run in 0..3 {
            let start = Instant::now();
            let frame = CompactMambaFrame::parse(&packed, &layout, 1024, [7; 32])?;
            let parse = start.elapsed();
            let restored = frame.expand(&device)?;
            let compact_total = start.elapsed();
            drop(restored);
            let start = Instant::now();
            let decoded = NemotronMambaSnapshot::from_bytes(&native)?;
            let native_parse = start.elapsed();
            let restored = decoded.restore_validated(&layout, 1024, [7; 32], &device)?;
            let native_total = start.elapsed();
            drop(restored);
            println!("run={run} compact_validate_ms={:.3} compact_expand_ms={:.3} compact_total_ms={:.3} native_parse_ms={:.3} native_expand_ms={:.3} native_total_ms={:.3}",
                parse.as_secs_f64() * 1000.0,
                (compact_total - parse).as_secs_f64() * 1000.0,
                compact_total.as_secs_f64() * 1000.0,
                native_parse.as_secs_f64() * 1000.0,
                (native_total - native_parse).as_secs_f64() * 1000.0,
                native_total.as_secs_f64() * 1000.0);
        }
        Ok(())
    }
}
