//! Device-side routed NVFP4 experts for eager Nemotron-H MoE layers.
//! The existing packed bytes and scale bits are stacked at load time; routing,
//! permutation, grouped GEMMs, and weighted combination stay on the GPU.

use super::NemotronMlp;
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "nemotron-expert-fused-quant")]
use std::sync::LazyLock;

#[cfg(feature = "nemotron-expert-fused-quant")]
static FUSED_QUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_EXPERT_FUSED_QUANT").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});
#[cfg(feature = "nemotron-expert-fused-quant")]
static FUSED_FINALIZE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_EXPERT_FUSED_FINALIZE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
        && std::env::var_os("XINFER_NEMOTRON_FUSED_MOE_COMBINE").as_deref()
            != Some(std::ffi::OsStr::new("0"))
});
#[cfg(feature = "nemotron-expert-fused-quant")]
static FUSED_RELU_QUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("XINFER_NEMOTRON_EXPERT_FUSED_RELU_QUANT").as_deref()
        == Some(std::ffi::OsStr::new("1"))
});

struct Projection {
    blocks: Tensor,
    scales: Tensor,
    global_scales: Tensor,
    input_scales: Tensor,
    swizzled_scales: Option<Tensor>,
}

impl Projection {
    #[cfg(feature = "nemotron-expert-fused-quant")]
    fn fused_quant_eligible(
        input: &Tensor,
        sorted: Option<(&Tensor, &Tensor)>,
        is_prefill: bool,
    ) -> Result<bool> {
        Ok(*FUSED_QUANT
            && is_prefill
            && input.dtype() == DType::BF16
            && input.is_contiguous()
            && input.storage_and_layout().1.start_offset() == 0
            && !attention_rs::nvfp4_online_scale()
            && attention_rs::cuda_utils::sm_version(input.device().as_cuda_device()?) == Some(121)
            && sorted.is_some_and(|(token_ids, expert_ids)| {
                token_ids.is_contiguous()
                    && expert_ids.is_contiguous()
                    && token_ids.storage_and_layout().1.start_offset() == 0
                    && expert_ids.storage_and_layout().1.start_offset() == 0
            }))
    }

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
        finalize: Option<(&Tensor, &Tensor)>,
        relu_squared: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "nemotron-expert-fused-quant")]
        if Self::fused_quant_eligible(input, sorted, is_prefill)? {
            let (sorted_ids, expert_ids) = sorted.ok_or_else(|| {
                candle_core::Error::Msg("fused expert quant requires grouped routes".into())
            })?;
            return super::expert_gemm::moe_gemm_nvfp4_fused_quant(
                input,
                &self.blocks,
                &self.scales,
                &self.global_scales,
                Some(&self.input_scales),
                &None,
                sorted_ids,
                expert_ids,
                ids.dim(1)?,
                is_prefill,
                self.swizzled_scales.as_ref(),
                finalize,
                relu_squared,
            );
        }
        if relu_squared {
            candle_core::bail!(
                "Nemotron fused ReLU quantization requires fused expert quantization"
            )
        }
        if finalize.is_some() {
            candle_core::bail!("Nemotron expert finalizer requires fused quantization")
        }
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
        let up = self
            .up
            .forward(xs, ids, sorted_ref, is_prefill, None, false)?;
        #[cfg(feature = "nemotron-expert-fused-quant")]
        let fuse_relu =
            *FUSED_RELU_QUANT && Projection::fused_quant_eligible(&up, sorted_ref, is_prefill)?;
        #[cfg(not(feature = "nemotron-expert-fused-quant"))]
        let fuse_relu = false;
        let activated = if fuse_relu {
            None
        } else {
            Some(super::relu_squared(&up)?)
        };
        let down_input = activated.as_ref().unwrap_or(&up);
        #[cfg(feature = "nemotron-expert-fused-quant")]
        let want_finalize = *FUSED_FINALIZE
            && topk <= 8
            && rows
                .checked_mul(hidden)
                .is_some_and(|n| n <= i32::MAX as usize)
            && weights.dtype() == DType::F32
            && weights.is_contiguous()
            && weights.storage_and_layout().1.start_offset() == 0
            && Projection::fused_quant_eligible(down_input, sorted_ref, is_prefill)?;
        #[cfg(not(feature = "nemotron-expert-fused-quant"))]
        let want_finalize = false;
        let prepared_shared = if want_finalize {
            Some(shared.forward(xs)?)
        } else {
            None
        };
        let finalize = prepared_shared.as_ref().and_then(|value| {
            (value.dtype() == DType::BF16
                && value.dims() == [rows, hidden]
                && value.is_contiguous()
                && value.storage_and_layout().1.start_offset() == 0)
                .then_some((value, weights))
        });
        let down = self
            .down
            .forward(down_input, ids, sorted_ref, is_prefill, finalize, fuse_relu)?;
        if finalize.is_some() {
            return Ok(down);
        }
        let down = down.reshape((rows, topk, hidden))?;
        let shared_output = if let Some(value) = prepared_shared {
            value
        } else {
            shared.forward(xs)?
        };
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
