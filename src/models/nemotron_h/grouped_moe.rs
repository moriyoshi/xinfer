//! Device-side routed NVFP4 experts for eager Nemotron-H MoE layers.
//! The existing packed bytes and scale bits are stacked at load time; routing,
//! permutation, grouped GEMMs, and weighted combination stay on the GPU.

use super::NemotronMlp;
use candle_core::{DType, Result, Tensor};

struct Projection {
    blocks: Tensor,
    scales: Tensor,
    global_scales: Tensor,
    input_scales: Tensor,
    swizzled_scales: Option<Tensor>,
}

impl Projection {
    fn from_experts<'a>(
        projections: impl Iterator<Item = &'a crate::models::layers::linear::LnNvfp4>,
    ) -> Result<Self> {
        let projections = projections.collect::<Vec<_>>();
        let first = projections[0];
        let blocks = Tensor::stack(
            &projections.iter().map(|p| &p.blocks).collect::<Vec<_>>(),
            0,
        )?;
        let scales = Tensor::stack(
            &projections.iter().map(|p| &p.scales).collect::<Vec<_>>(),
            0,
        )?;
        let swizzled_scales = if projections
            .iter()
            .all(|p| p.weight_scale_swizzled.is_some())
        {
            Some(Tensor::stack(
                &projections
                    .iter()
                    .map(|p| p.weight_scale_swizzled.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                0,
            )?)
        } else {
            None
        };
        let count = projections.len();
        let global_scales = Tensor::from_vec(
            projections
                .iter()
                .map(|p| p.global_scale)
                .collect::<Vec<_>>(),
            (count,),
            first.blocks.device(),
        )?;
        let input_scales = Tensor::from_vec(
            projections
                .iter()
                .map(|p| p.input_scale)
                .collect::<Vec<_>>(),
            (count,),
            first.blocks.device(),
        )?;
        Ok(Self {
            blocks,
            scales,
            global_scales,
            input_scales,
            swizzled_scales,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        ids: &Tensor,
        sorted: Option<(&Tensor, &Tensor)>,
        is_prefill: bool,
    ) -> Result<Tensor> {
        attention_rs::moe::moe_gemm_nvfp4(
            input,
            &self.blocks,
            &self.scales,
            &self.global_scales,
            Some(&self.input_scales),
            None,
            ids,
            sorted,
            is_prefill,
            None,
            self.swizzled_scales.as_ref(),
            None,
        )
    }
}

pub(super) struct GroupedNvfp4Experts {
    up: Projection,
    down: Projection,
}

impl GroupedNvfp4Experts {
    pub(super) fn from_experts(experts: &[NemotronMlp]) -> Result<Option<Self>> {
        let hardware_shape = |projection: &crate::models::layers::linear::LnNvfp4| {
            let shape = projection.blocks.dims();
            shape.len() == 2 && shape[0].is_multiple_of(32) && shape[1].is_multiple_of(16)
        };
        if experts.is_empty()
            || experts.iter().any(|expert| {
                expert
                    .up
                    .as_nvfp4()
                    .is_none_or(|projection| !hardware_shape(projection))
                    || expert
                        .down
                        .as_nvfp4()
                        .is_none_or(|projection| !hardware_shape(projection))
            })
        {
            return Ok(None);
        }
        let up =
            Projection::from_experts(experts.iter().map(|expert| expert.up.as_nvfp4().unwrap()))?;
        let down =
            Projection::from_experts(experts.iter().map(|expert| expert.down.as_nvfp4().unwrap()))?;
        Ok(Some(Self { up, down }))
    }

    pub(super) fn forward(
        &self,
        xs: &Tensor,
        shared: &NemotronMlp,
        weights: &Tensor,
        ids: &Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let rows = xs.dim(0)?;
        let topk = ids.dim(1)?;
        let hidden = xs.dim(1)?;
        let sorted = if is_prefill {
            let (expert_ids, token_ids) =
                super::route_grouping::sort_expert_assignments(ids, self.up.blocks.dim(0)?)?;
            Some((token_ids, expert_ids))
        } else {
            None
        };
        let sorted_ref = sorted.as_ref().map(|(token, expert)| (token, expert));
        let up = self.up.forward(xs, ids, sorted_ref, is_prefill)?;
        let activated = up.relu()?.sqr()?;
        let down = self.down.forward(&activated, ids, sorted_ref, is_prefill)?;
        let down = down.reshape((rows, topk, hidden))?;
        let shared_output = shared.forward(xs)?;
        if let Some(combined) =
            super::route_grouping::combine_routed(&down, &shared_output, weights)?
        {
            return Ok(combined);
        }
        // Match the original path's F32 route scaling after the BF16 expert
        // output. The sum is entirely device-side and uses no host routing.
        let routed = down
            .to_dtype(DType::F32)?
            .broadcast_mul(&weights.unsqueeze(2)?)?
            .sum(1)?;
        routed
            .broadcast_add(&shared_output.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())
    }
}
