//! Experimental exact-token sparse attention for selected Qwen3 query heads.
//! This deliberately computes on the CPU after the native attention kernel.
//! It is a quality probe, not a serving fast path.

use super::attention::AttentionProbe;
use candle_core::{DType, Result, Tensor};
use std::{collections::HashSet, sync::Arc};

/// One query row/head's retained, already-rotated K/V. `positions` are the
/// original logical positions; they never describe a condensed cache. Entries
/// before `exact_start` may be an arbitrary sorted selection. Every position
/// from `exact_start` through the query position must appear exactly once,
/// including the exact prepared-prefix tail and visible suffix. `prefix_end`
/// marks the boundary between that tail and the suffix. The caller owns the
/// evidence that supplied K/V bytes came from those positions and KV head.
pub struct SparseAttentionHead {
    pub row: usize,
    pub query_head: usize,
    pub kv_head: usize,
    pub prefix_end: usize,
    pub exact_start: usize,
    pub positions: Vec<usize>,
    pub keys: Tensor,   // [positions.len(), head_width]
    pub values: Tensor, // [positions.len(), head_width]
}

/// Called after native attention has populated its KV cache. Return no rows
/// for layers or queries that should use the native context. A callback can
/// supply several rows/heads from a batch. Errors abort the current forward;
/// the caller must discard its partially updated KV cache on failure.
pub type SparseAttentionProvider =
    Arc<dyn Fn(&AttentionProbe) -> Result<Vec<SparseAttentionHead>> + Send + Sync>;

fn logical_positions(probe: &AttentionProbe) -> Result<Vec<usize>> {
    let positions = probe.positions.flatten_all()?;
    match positions.dtype() {
        DType::I64 => positions
            .to_vec1::<i64>()?
            .into_iter()
            .map(|position| {
                usize::try_from(position).map_err(|_| {
                    candle_core::Error::Msg("negative sparse attention query position".into())
                })
            })
            .collect(),
        DType::U32 => Ok(positions
            .to_vec1::<u32>()?
            .into_iter()
            .map(|position| position as usize)
            .collect()),
        other => candle_core::bail!("unsupported sparse attention position dtype {other:?}"),
    }
}

/// Replace only the requested row/head contexts. All other heads retain their
/// native values. The supplied K/V participate in one stable softmax, so old
/// prefix, exact tail and suffix share the same normalization mass.
pub(crate) fn apply_sparse_attention_heads(
    probe: &AttentionProbe,
    heads: Vec<SparseAttentionHead>,
    sliding_window: Option<usize>,
    softcapping: Option<f64>,
) -> Result<Option<Tensor>> {
    if heads.is_empty() {
        return Ok(None);
    }
    if sliding_window.is_some() || softcapping.is_some() {
        candle_core::bail!(
            "sparse attention probe requires full causal attention without logit softcap"
        )
    }
    if probe.query_heads == 0
        || probe.kv_heads == 0
        || probe.head_width == 0
        || probe.query_heads % probe.kv_heads != 0
        || !probe.score_scale.is_finite()
        || probe.score_scale <= 0.0
    {
        candle_core::bail!("invalid sparse attention head geometry or scale")
    }
    let rows = probe.query.dim(0)?;
    let shape = (rows, probe.query_heads, probe.head_width);
    if probe.query.dims() != [shape.0, shape.1, shape.2]
        || probe.context.dims() != [shape.0, shape.1, shape.2]
        || probe.context.dtype() != probe.dtype
    {
        candle_core::bail!("sparse attention query/context shape or dtype mismatch")
    }
    let query_positions = logical_positions(probe)?;
    if query_positions.len() != rows {
        candle_core::bail!("sparse attention position count differs from query rows")
    }
    let query = probe
        .query
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut result = probe
        .context
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut seen = HashSet::with_capacity(heads.len());
    for head in heads {
        if head.row >= rows || head.query_head >= probe.query_heads {
            candle_core::bail!("sparse attention row or query head is out of range")
        }
        if !seen.insert((head.row, head.query_head)) {
            candle_core::bail!("duplicate sparse attention row/head replacement")
        }
        let expected_kv_head = head.query_head / (probe.query_heads / probe.kv_heads);
        if head.kv_head != expected_kv_head {
            candle_core::bail!("sparse attention KV head does not match GQA query head")
        }
        let query_position = query_positions[head.row];
        if head.exact_start > head.prefix_end || head.prefix_end > query_position {
            candle_core::bail!("invalid sparse attention exact tail or prefix boundary")
        }
        let count = head.positions.len();
        if head.keys.dims() != [count, probe.head_width]
            || head.values.dims() != [count, probe.head_width]
            || head.keys.dtype() != probe.dtype
            || head.values.dtype() != probe.dtype
        {
            candle_core::bail!("sparse attention K/V shape or dtype mismatch")
        }
        let exact_count = query_position
            .checked_sub(head.exact_start)
            .and_then(|remaining| remaining.checked_add(1))
            .ok_or_else(|| {
                candle_core::Error::Msg("sparse attention exact range overflow".into())
            })?;
        if count < exact_count {
            candle_core::bail!("sparse attention exact tail or suffix has missing positions")
        }
        let old_count = count - exact_count;
        if head.positions[..old_count]
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || head.positions[..old_count]
                .iter()
                .any(|&position| position >= head.exact_start)
            || head.positions[old_count..]
                .iter()
                .enumerate()
                .any(|(offset, &position)| position != head.exact_start + offset)
        {
            candle_core::bail!("sparse attention positions must preserve sorted logical indices and a complete exact tail/suffix")
        }
        let keys = head
            .keys
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let values = head
            .values
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let width = probe.head_width;
        let offset = (head.row * probe.query_heads + head.query_head) * width;
        let q = &query[offset..offset + width];
        let scores = keys
            .chunks_exact(width)
            .map(|key| {
                q.iter()
                    .zip(key)
                    .map(|(&left, &right)| f64::from(left) * f64::from(right))
                    .sum::<f64>()
                    * f64::from(probe.score_scale)
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let masses = scores
            .iter()
            .map(|&score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = masses.iter().sum::<f64>();
        if !denominator.is_finite() || denominator == 0.0 {
            candle_core::bail!("invalid sparse attention softmax mass")
        }
        for channel in 0..width {
            result[offset + channel] = (masses
                .iter()
                .enumerate()
                .map(|(token, &mass)| mass * f64::from(values[token * width + channel]))
                .sum::<f64>()
                / denominator) as f32;
        }
    }
    Ok(Some(
        Tensor::from_vec(result, shape, probe.context.device())?.to_dtype(probe.dtype)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn probe() -> Result<AttentionProbe> {
        Ok(AttentionProbe {
            layer_index: 4,
            positions: Tensor::from_vec(vec![3i64], (1,), &Device::Cpu)?,
            sequence_ids: Some(vec![7]),
            seqlens: None,
            is_prefill: false,
            query: Tensor::from_vec(vec![1f32, 0., 0., 1.], (1, 2, 2), &Device::Cpu)?,
            context: Tensor::from_vec(vec![0f32, 0., 7., 9.], (1, 2, 2), &Device::Cpu)?,
            effective_context: None,
            query_heads: 2,
            kv_heads: 1,
            head_width: 2,
            dtype: DType::F32,
            score_scale: 1.0,
        })
    }

    fn head(positions: Vec<usize>) -> Result<SparseAttentionHead> {
        let source_keys = [[1f32, 0.], [0., 1.], [1., 1.], [-1., 0.]];
        let source_values = [[10f32, 0.], [0., 10.], [0., 20.], [30., 40.]];
        let count = positions.len();
        let keys = positions
            .iter()
            .flat_map(|&position| source_keys[position])
            .collect::<Vec<_>>();
        let values = positions
            .iter()
            .flat_map(|&position| source_values[position])
            .collect::<Vec<_>>();
        Ok(SparseAttentionHead {
            row: 0,
            query_head: 0,
            kv_head: 0,
            prefix_end: 3,
            exact_start: 2,
            positions,
            keys: Tensor::from_vec(keys, (count, 2), &Device::Cpu)?,
            values: Tensor::from_vec(values, (count, 2), &Device::Cpu)?,
        })
    }

    #[test]
    fn sparse_head_uses_one_mass_and_keeps_other_head() -> Result<()> {
        let probe = probe()?;
        let output = apply_sparse_attention_heads(&probe, vec![head(vec![0, 2, 3])?], None, None)?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let weights = [1f32.exp(), 1f32.exp(), (-1f32).exp()];
        let mass = weights.iter().sum::<f32>();
        let expected = [
            (weights[0] * 10. + weights[2] * 30.) / mass,
            (weights[1] * 20. + weights[2] * 40.) / mass,
            7.,
            9.,
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert_eq!(
            probe.context.flatten_all()?.to_vec1::<f32>()?,
            [0., 0., 7., 9.]
        );
        Ok(())
    }

    #[test]
    fn rejects_condensed_or_incomplete_logical_positions() -> Result<()> {
        let probe = probe()?;
        for positions in [vec![0, 3], vec![0, 3, 2], vec![0, 2, 2, 3]] {
            assert!(
                apply_sparse_attention_heads(&probe, vec![head(positions)?], None, None,).is_err()
            );
        }
        let mut wrong_head = head(vec![0, 2, 3])?;
        wrong_head.kv_head = 1;
        assert!(apply_sparse_attention_heads(&probe, vec![wrong_head], None, None).is_err());
        assert!(apply_sparse_attention_heads(
            &probe,
            vec![head(vec![0, 2, 3])?, head(vec![0, 2, 3])?],
            None,
            None,
        )
        .is_err());
        assert!(
            apply_sparse_attention_heads(&probe, vec![head(vec![0, 2, 3])?], Some(128), None,)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn long_prefix_retains_scattered_indices_and_requires_full_exact_suffix() -> Result<()> {
        let mut probe = probe()?;
        probe.positions = Tensor::from_vec(vec![2048i64], (1,), &Device::Cpu)?;
        let positions = [0usize, 17, 128]
            .into_iter()
            .chain(2016..=2048)
            .collect::<Vec<_>>();
        let count = positions.len();
        let record = || -> Result<SparseAttentionHead> {
            Ok(SparseAttentionHead {
                row: 0,
                query_head: 0,
                kv_head: 0,
                prefix_end: 2048,
                exact_start: 2016,
                positions: positions.clone(),
                keys: Tensor::from_vec(vec![0f32; count * 2], (count, 2), &Device::Cpu)?,
                values: Tensor::from_vec(vec![1f32; count * 2], (count, 2), &Device::Cpu)?,
            })
        };
        assert!(apply_sparse_attention_heads(&probe, vec![record()?], None, None)?.is_some());
        let mut missing = record()?;
        missing.positions.remove(20);
        missing.keys = Tensor::zeros((count - 1, 2), DType::F32, &Device::Cpu)?;
        missing.values = Tensor::zeros((count - 1, 2), DType::F32, &Device::Cpu)?;
        assert!(apply_sparse_attention_heads(&probe, vec![missing], None, None).is_err());
        Ok(())
    }
}
