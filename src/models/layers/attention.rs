use crate::models::layers::distributed::{
    kv_head_shard, shard, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::linear::{LinearX, LnNvfp4};
use crate::models::layers::others::{rms_norm, rms_norm_sharded, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::sparse_attention::{
    apply_sparse_attention_heads, SparseAttentionProvider,
};
use crate::models::layers::{isq_high_precision_quant, VarBuilderX};
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Result, Tensor, D};
use either::Either;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

enum QkvProjection {
    QueryOnly(TensorParallelColumnLinear),
    Separate {
        q_proj: TensorParallelColumnLinear,
        k_proj: TensorParallelColumnLinear,
        v_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
    PackedViews(MergedParallelColumnLinear),
}

/// Opt-in observation of the exact query sent to paged attention and its
/// per-head result, before the output gate and output projection. The query,
/// context, and positions remain on their original device; an observer that
/// needs host bytes must copy them explicitly. Rows are in the input's packed
/// token order, with `seqlens` delimiting sequences during batched prefill.
/// Retaining these tensors also retains their GPU storage.
#[derive(Clone)]
pub struct AttentionProbe {
    pub layer_index: usize,
    pub positions: Tensor,
    pub sequence_ids: Option<Vec<usize>>,
    pub seqlens: Option<Vec<u32>>,
    pub is_prefill: bool,
    pub query: Tensor,   // [tokens, local query heads, head width]
    pub context: Tensor, // [tokens, local query heads, head width]
    /// Present only when an experimental sparse replacement was applied.
    /// `context` always remains the native kernel result.
    pub effective_context: Option<Tensor>,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_width: usize,
    pub dtype: DType,
    pub score_scale: f32,
}

pub type AttentionProbeObserver = Arc<dyn Fn(AttentionProbe) -> Result<()> + Send + Sync>;

struct AttentionHooks<'a> {
    layer_index: usize,
    observer: Option<&'a AttentionProbeObserver>,
    sparse_provider: Option<&'a SparseAttentionProvider>,
}

#[cfg(feature = "flash")]
struct SharedKvSplitKWorkspace {
    tensor: Tensor,
    max_batch: usize,
}

#[cfg(feature = "flash")]
impl SharedKvSplitKWorkspace {
    fn new(
        max_batch: usize,
        heads: usize,
        width: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if !(1..=64).contains(&max_batch) || heads == 0 || width == 0 {
            candle_core::bail!(
                "shared-KV split-K requires batch capacity 1..=64 and nonzero heads/width"
            )
        }
        let elements = max_batch
            .checked_mul(heads)
            .and_then(|v| v.checked_mul(attention_rs::flash::NUM_SPLITS as usize))
            .and_then(|v| width.checked_add(2).and_then(|width| v.checked_mul(width)))
            .ok_or_else(|| {
                candle_core::Error::Msg("shared-KV split-K workspace overflow".into())
            })?;
        Ok(Self {
            tensor: Tensor::zeros(elements, DType::F32, device)?,
            max_batch,
        })
    }

    fn for_batch(&self, batch: usize) -> Option<&Tensor> {
        (batch > 0 && batch <= self.max_batch).then_some(&self.tensor)
    }
}

pub struct Attention {
    qkv_proj: QkvProjection,
    o_proj: TensorParallelRowLinear,
    q_norm: Option<NormX>,
    k_norm: Option<NormX>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    attn: PagedAttention,
    score_scale: f32,
    sliding_window: Option<usize>,
    softcapping: Option<f64>,
    dtype: DType,
    full_dim_qk_norm: bool,
    is_qvar_builder: bool,
    qk_l2_norm: bool,
    promote_qk_to_f32: bool,
    v_norm_eps: Option<f64>,
    nemotron_attention_gemm: bool,
    gemma4_fused_qk_rope: bool,
    #[cfg(feature = "flash")]
    shared_kv_splitk: Option<SharedKvSplitKWorkspace>,
}

impl Attention {
    fn try_pack_gemma4_nvfp4_qkv(
        q_proj: &TensorParallelColumnLinear,
        k_proj: &TensorParallelColumnLinear,
        v_proj: &TensorParallelColumnLinear,
    ) -> Result<Option<MergedParallelColumnLinear>> {
        let (Some(q), Some(k), Some(v)) = (q_proj.as_nvfp4(), k_proj.as_nvfp4(), v_proj.as_nvfp4())
        else {
            return Ok(None);
        };
        // A single activation quantization and GEMM must use the same scalar
        // calibration as each original projection. Other checkpoints retain
        // their separate projections if either scalar differs.
        if [k, v].iter().any(|p| {
            p.input_scale.to_bits() != q.input_scale.to_bits()
                || p.global_scale.to_bits() != q.global_scale.to_bits()
                || p.bias.is_some()
        }) || q.bias.is_some()
        {
            return Ok(None);
        }
        let local_dims = [q.blocks.dim(0)?, k.blocks.dim(0)?, v.blocks.dim(0)?];
        if local_dims.iter().any(|dim| !dim.is_multiple_of(32))
            || k.blocks.dim(1)? != q.blocks.dim(1)?
            || v.blocks.dim(1)? != q.blocks.dim(1)?
            || k.scales.dim(1)? != q.scales.dim(1)?
            || v.scales.dim(1)? != q.scales.dim(1)?
        {
            return Ok(None);
        }

        let blocks = Tensor::cat(&[&q.blocks, &k.blocks, &v.blocks], 0)?;
        let scales = Tensor::cat(&[&q.scales, &k.scales, &v.scales], 0)?;
        #[cfg(feature = "cuda")]
        let weight_scale_swizzled = if q.weight_scale_swizzled.is_some()
            && k.weight_scale_swizzled.is_some()
            && v.weight_scale_swizzled.is_some()
        {
            Some(attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                &scales,
            )?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let weight_scale_swizzled = None;
        let packed = LinearX::LnNvfp4(LnNvfp4 {
            blocks,
            scales,
            global_scale: q.global_scale,
            input_scale: q.input_scale,
            bias: None,
            weight_scale_swizzled,
            resident_scales: q.resident_scales.clone(),
        });
        MergedParallelColumnLinear::from_packed_local_qlinear(packed, None, local_dims.to_vec())
            .map(Some)
    }

    fn normalize_sharded_2d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            candle_core::bail!("unexpected shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            );
        }

        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name,
            d0,
            d1,
            local,
            global_dim1
        );
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: candle_nn::var_builder::Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}), expects local {} or global {}",
            name,
            d0,
            local,
            global_dim
        );
    }

    fn load_sharded_bias(
        vb: &VarBuilderX,
        out_dim: usize,
        shard: candle_nn::var_builder::Shard,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let bias = match &vb.0 {
            Either::Left(inner) => {
                inner.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32)
            }
            Either::Right(_) => return Ok(None),
        };
        let Ok(bias) = bias else {
            return Ok(None);
        };
        let bias = Self::normalize_sharded_1d(bias, shard, out_dim, "bias")?;
        if bias.dtype() != dtype {
            Ok(Some(bias.to_dtype(dtype)?))
        } else {
            Ok(Some(bias))
        }
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilderX,
        out_dim: usize,
        in_dim: usize,
        shard: candle_nn::var_builder::Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.has_key("weight_scale") && !vb.has_key("weight_scale_inv") {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb
            .get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::F8E4M3)
            .or_else(|_| vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8))
        {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;

        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(scale) => scale,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(scale) => scale,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;

        Ok(Some((weight, weight_scale)))
    }

    fn try_load_packed_qkv(
        vb: &VarBuilderX,
        hidden_size: usize,
        q_out_dim: usize,
        kv_out_dim: usize,
        attention_bias: bool,
        comm: Rc<Comm>,
        kv_shard: candle_nn::var_builder::Shard,
        dtype: DType,
        quant_cfg: &Option<crate::utils::config::QuantConfig>,
        quant: &Option<String>,
        k_eq_v: bool,
    ) -> Result<Option<QkvProjection>> {
        if vb.is_qvar_builder() || quant.is_some() {
            return Ok(None);
        }

        let q_shard = shard(0, comm.rank(), comm.world_size());
        let q_vb = vb.pp("q_proj");
        let k_vb = vb.pp("k_proj");
        let v_vb = if k_eq_v {
            vb.pp("k_proj")
        } else {
            vb.pp("v_proj")
        };

        let is_fp8_quant = quant_cfg
            .as_ref()
            .map(|cfg| cfg.quant_method == "fp8")
            .unwrap_or(false);
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                candle_core::bail!("LnFp8: weight_block_size must be configured for packed qkv");
            };
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }

            let Some((q_weight, q_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &q_vb,
                q_out_dim,
                hidden_size,
                q_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((k_weight, k_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &k_vb,
                kv_out_dim,
                hidden_size,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((v_weight, v_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &v_vb,
                kv_out_dim,
                hidden_size,
                kv_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };

            let local_q = q_weight.dim(0)?;
            let local_k = k_weight.dim(0)?;
            let local_v = v_weight.dim(0)?;
            let by = block_size[0];
            let q_global_start = q_shard.rank * local_q;
            let k_global_start = q_out_dim + kv_shard.rank * local_k;
            let v_global_start = q_out_dim + kv_out_dim + kv_shard.rank * local_v;
            if q_global_start % by != 0 || k_global_start % by != 0 || v_global_start % by != 0 {
                return Ok(None);
            }

            let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;
            let packed_scale = Tensor::cat(&[&q_scale, &k_scale, &v_scale], 0)?;
            let packed_bias = if attention_bias {
                let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
                let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
                let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
                match (q_bias, k_bias, v_bias) {
                    (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                    (None, None, None) => None,
                    _ => return Ok(None),
                }
            } else {
                None
            };

            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;

            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;

            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                packed_bias,
                block_size,
                sm_version,
                vec![local_q, local_k, local_v],
            )?;
            return Ok(Some(QkvProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let q_weight =
            q_vb.get_with_hints_dtype((q_out_dim, hidden_size), "weight", q_shard, dtype)?;
        let k_weight =
            k_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;
        let v_weight =
            v_vb.get_with_hints_dtype((kv_out_dim, hidden_size), "weight", kv_shard, dtype)?;

        let local_q = q_weight.dim(0)?;
        let local_k = k_weight.dim(0)?;
        let local_v = v_weight.dim(0)?;
        let packed_weight = Tensor::cat(&[&q_weight, &k_weight, &v_weight], 0)?;

        let packed_bias = if attention_bias {
            let q_bias = Self::load_sharded_bias(&q_vb, q_out_dim, q_shard, dtype)?;
            let k_bias = Self::load_sharded_bias(&k_vb, kv_out_dim, kv_shard, dtype)?;
            let v_bias = Self::load_sharded_bias(&v_vb, kv_out_dim, kv_shard, dtype)?;
            match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[&qb, &kb, &vb], 0)?),
                (None, None, None) => None,
                _ => return Ok(None),
            }
        } else {
            None
        };

        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            packed_bias,
            vec![local_q, local_k, local_v],
            &None,
        )?;
        Ok(Some(QkvProjection::Packed(merged)))
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_options(
            vb,
            comm,
            config,
            attention_scale,
            sliding_window,
            dtype,
            false,
            false,
        )
    }

    pub fn new_with_options(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
        k_eq_v: bool,
        qk_l2_norm: bool,
    ) -> Result<Self> {
        Self::new_with_options_impl(
            vb,
            comm,
            config,
            attention_scale,
            sliding_window,
            dtype,
            k_eq_v,
            qk_l2_norm,
            false,
        )
    }

    /// Gemma 4 shared layers have no usable K/V projection weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options_shared_kv(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
        k_eq_v: bool,
        qk_l2_norm: bool,
    ) -> Result<Self> {
        Self::new_with_options_impl(
            vb,
            comm,
            config,
            attention_scale,
            sliding_window,
            dtype,
            k_eq_v,
            qk_l2_norm,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_options_impl(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
        k_eq_v: bool,
        qk_l2_norm: bool,
        query_only: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);
        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
            ("q_norm", "attn_q_norm"),
            ("k_norm", "attn_k_norm"),
        ]
        .iter()
        .cloned()
        .collect();

        let is_qvar_builder = vb.is_qvar_builder();
        let arch = config.architectures.as_ref().unwrap()[0].clone();
        let is_qwen35_or_next = matches!(
            arch.as_str(),
            "Qwen3_5ForCausalLM"
                | "Qwen3_5ForConditionalGeneration"
                | "Qwen3_5MoeForCausalLM"
                | "Qwen3_5MoeForConditionalGeneration"
                | "Qwen3NextForCausalLM"
                | "Qwen3NextForConditionalGeneration"
                | "Qwen4ExpForConditionalGeneration"
                | "Qwen4ExpForCausalLM"
        );
        let attention_bias = if is_qwen35_or_next {
            config.qkv_bias.or(config.attention_bias).unwrap_or(false)
        } else {
            config.qkv_bias.or(config.attention_bias).unwrap_or(true)
        };
        let attn_output_gate = if is_qwen35_or_next {
            config.attn_output_gate.unwrap_or(true)
        } else {
            config.attn_output_gate.unwrap_or(false)
        };
        let q_out_dim = num_heads * head_dim * if attn_output_gate { 2 } else { 1 };
        let is_gemma = arch == "Gemma3ForConditionalGeneration".to_string()
            || arch == "Gemma3ForCausalLM".to_string();
        let is_mlx_nvfp4 = config
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.is_mlx_nvfp4);
        // Qwen3.5/Qwen3Next HF weights use Gemma-style +1 weight semantics for
        // q/k norms. MLX-sanitized NVFP4 checkpoints store the actual weights.
        let qk_norm_add_one = is_gemma || (is_qwen35_or_next && !is_qvar_builder && !is_mlx_nvfp4);

        let world_size = comm.world_size();
        let attention_heads = num_heads / world_size;
        let (kv_heads, kv_shard) = kv_head_shard(num_kv_heads, comm.rank(), world_size)?;
        #[cfg(feature = "cuda")]
        let packed_gemma4_nvfp4 = matches!(
            arch.as_str(),
            "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM"
        ) && !attention_bias
            && !attn_output_gate
            && dtype == DType::BF16
            && cfg!(feature = "cutlass")
            && vb.device().is_cuda()
            && attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?).unwrap_or(0)
                >= 100
            && config
                .quantization_config
                .as_ref()
                .is_some_and(|cfg| cfg.quant_method == "nvfp4")
            && std::env::var_os("XINFER_GEMMA4_PACKED_QKV_NVFP4").is_some_and(|value| value == "1");
        #[cfg(not(feature = "cuda"))]
        let packed_gemma4_nvfp4 = false;

        let qkv_proj = if query_only {
            QkvProjection::QueryOnly(TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                q_out_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map["q_proj"])
                } else {
                    vb.pp("q_proj")
                },
                comm.clone(),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?)
        } else if let Some(packed) = Self::try_load_packed_qkv(
            &vb,
            hidden_size,
            q_out_dim,
            num_kv_heads * head_dim,
            attention_bias,
            comm.clone(),
            kv_shard,
            dtype,
            &config.quantization_config,
            &config.quant,
            k_eq_v,
        )? {
            packed
        } else {
            let q_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                q_out_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map["q_proj"])
                } else {
                    vb.pp("q_proj")
                },
                comm.clone(),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let k_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map["k_proj"])
                } else {
                    vb.pp("k_proj")
                },
                kv_shard,
                &config.quantization_config,
                &config.quant,
                dtype,
            )?;
            let high_precision_quant = config
                .quant
                .as_deref()
                .map(|quant| isq_high_precision_quant(quant).to_string());
            let v_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_size,
                num_kv_heads * head_dim,
                attention_bias,
                if is_qvar_builder {
                    vb.pp(key_map[if k_eq_v { "k_proj" } else { "v_proj" }])
                } else {
                    vb.pp(if k_eq_v { "k_proj" } else { "v_proj" })
                },
                kv_shard,
                &config.quantization_config,
                if config.quant.is_some() && config.quantization_config.is_none() {
                    &high_precision_quant
                } else {
                    &None
                },
                dtype,
            )?;
            if packed_gemma4_nvfp4 {
                if let Some(packed) = Self::try_pack_gemma4_nvfp4_qkv(&q_proj, &k_proj, &v_proj)? {
                    QkvProjection::PackedViews(packed)
                } else {
                    QkvProjection::Separate {
                        q_proj,
                        k_proj,
                        v_proj,
                    }
                }
            } else {
                QkvProjection::Separate {
                    q_proj,
                    k_proj,
                    v_proj,
                }
            }
        };

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                vb.pp("o_proj")
            },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let norm_dtype = if is_qvar_builder
            || config.quant.is_some()
            || config.quantization_config.is_some()
            || config.is_f16_mode
        {
            DType::F32
        } else {
            dtype
        };
        let q_norm_vb = if is_qvar_builder {
            vb.pp(key_map["q_norm"])
        } else {
            vb.pp("q_norm")
        };
        let k_norm_vb = if is_qvar_builder {
            vb.pp(key_map["k_norm"])
        } else {
            vb.pp("k_norm")
        };

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            q_norm_vb.clone(),
            norm_dtype,
            qk_norm_add_one,
        );
        let k_norm = if query_only {
            None
        } else {
            Some(rms_norm(
                head_dim,
                config.rms_norm_eps,
                k_norm_vb.clone(),
                norm_dtype,
                qk_norm_add_one,
            ))
        };

        let (q_norm, k_norm, full_dim_qk_norm) = if query_only {
            if let Ok(q_norm) = q_norm {
                (Some(q_norm), None, false)
            } else {
                let q_shard = shard(0, comm.rank(), comm.world_size());
                match rms_norm_sharded(
                    num_heads * head_dim,
                    config.rms_norm_eps,
                    q_norm_vb,
                    norm_dtype,
                    qk_norm_add_one,
                    q_shard,
                ) {
                    Ok(q_norm) => (Some(q_norm), None, true),
                    Err(_) => (None, None, false),
                }
            }
        } else if q_norm.is_ok() && k_norm.as_ref().is_some_and(Result::is_ok) {
            (Some(q_norm.unwrap()), Some(k_norm.unwrap().unwrap()), false)
        } else {
            let q_shard = shard(0, comm.rank(), comm.world_size());
            let q_full = rms_norm_sharded(
                num_heads * head_dim,
                config.rms_norm_eps,
                q_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                q_shard,
            );
            let k_full = rms_norm_sharded(
                num_kv_heads * head_dim,
                config.rms_norm_eps,
                k_norm_vb,
                norm_dtype,
                qk_norm_add_one,
                kv_shard,
            );
            if q_full.is_ok() && k_full.is_ok() {
                (Some(q_full.unwrap()), Some(k_full.unwrap()), true)
            } else {
                (None, None, false)
            }
        };

        let is_gemma4 = arch == "Gemma4ForConditionalGeneration" || arch == "Gemma4ForCausalLM";
        let v_norm_eps = if is_gemma4 {
            Some(config.rms_norm_eps)
        } else {
            None
        };

        let score_scale = attention_scale.unwrap_or(1. / (head_dim as f32).sqrt());
        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn_output_gate,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                score_scale,
                Some(kv_heads),
                sliding_window,
                vb.device().clone(),
                None,
                config.kvcache_dtype.is_fp8_keys(),
            )?,
            score_scale,
            sliding_window,
            softcapping: config.attn_logit_softcapping,
            dtype,
            full_dim_qk_norm,
            is_qvar_builder,
            qk_l2_norm,
            promote_qk_to_f32: is_qvar_builder || config.higher_precision_required() || qk_l2_norm,
            v_norm_eps,
            nemotron_attention_gemm: false,
            gemma4_fused_qk_rope: false,
            #[cfg(feature = "flash")]
            shared_kv_splitk: None,
        })
    }

    /// Prepare shared-KV decode after all model weights have been loaded.
    /// Keeping scratch allocation out of weight loading preserves its layout.
    #[cfg(feature = "flash")]
    pub(crate) fn prepare_gemma4_shared_kv_splitk(
        &mut self,
        device: &candle_core::Device,
    ) -> Result<()> {
        if !matches!(self.qkv_proj, QkvProjection::QueryOnly(_))
            || self.sliding_window.is_some()
            || !device.is_cuda()
            || std::env::var_os("XINFER_GEMMA4_SHARED_KV_SPLITK").as_deref()
                != Some(std::ffi::OsStr::new("1"))
        {
            return Ok(());
        }
        let capacity = match std::env::var("XINFER_GEMMA4_SHARED_KV_SPLITK_MAX_BATCH") {
            Ok(value) => value.parse::<usize>().map_err(|_| {
                candle_core::Error::Msg("invalid shared-KV split-K batch capacity".into())
            })?,
            Err(std::env::VarError::NotPresent) => 8,
            Err(_) => candle_core::bail!("invalid shared-KV split-K batch capacity"),
        };
        // Allocate before graph capture. Larger batches retain single-pass.
        self.shared_kv_splitk = Some(SharedKvSplitKWorkspace::new(
            capacity,
            self.num_heads,
            self.head_dim,
            device,
        )?);
        Ok(())
    }

    /// Enables the experimental Nemotron-only BF16 paged prefill backend.
    /// The runtime opt-in remains `XINFER_NEMOTRON_ATTENTION_GEMM=1`.
    pub fn with_nemotron_attention_gemm(mut self) -> Self {
        self.nemotron_attention_gemm = true;
        self
    }

    /// Enables the Gemma 4 BF16 Q/K norm and RoPE CUDA fusion.
    pub fn with_gemma4_fused_qk_rope(mut self, enabled: bool) -> Self {
        self.gemma4_fused_qk_rope = enabled;
        self
    }

    fn fused_gemma4_qk_rope(
        &self,
        q: &Tensor,
        k: Option<&Tensor>,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Option<Tensor>)>> {
        #[cfg(feature = "cuda")]
        if self.gemma4_fused_qk_rope
            && self.promote_qk_to_f32
            && !self.full_dim_qk_norm
            && !self.qk_l2_norm
            && self.dtype == DType::BF16
        {
            if let (Some(q_norm), Some(rope)) = (&self.q_norm, rotary_emb) {
                if let (Some((q_weight, q_eps)), Some((cos, sin, interleaved))) =
                    (q_norm.f32_rms_weight_eps(), rope.full_f32_tables())
                {
                    let k_params = if let Some(k) = k {
                        self.k_norm
                            .as_ref()
                            .and_then(NormX::f32_rms_weight_eps)
                            .map(|(weight, eps)| (k, weight, eps))
                    } else {
                        None
                    };
                    if k.is_none() || k_params.is_some() {
                        if let Some(result) = crate::models::layers::gemma4_qk_rope::forward(
                            q,
                            k_params,
                            &q_weight,
                            q_eps,
                            cos,
                            sin,
                            positions,
                            interleaved,
                        )? {
                            return Ok(Some(result));
                        }
                    }
                }
            }
        }
        let _ = (q, k, rotary_emb, positions);
        Ok(None)
    }

    /// Promote Q/K to F32 when needed and apply per-token or per-head RMSNorm before RoPE.
    fn prepare_qk_for_rope(
        &self,
        q: Tensor,
        k: Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (q, k) = if self.promote_qk_to_f32 && q.dtype() != DType::F32 {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?)
        } else {
            (q, k)
        };

        let (q, k) = if self.q_norm.is_some() && self.k_norm.is_some() {
            if self.full_dim_qk_norm {
                let q_2d = q.reshape((seq_len, self.num_heads * self.head_dim))?;
                let k_2d = k.reshape((seq_len, self.num_kv_heads * self.head_dim))?;
                let q_2d = self.q_norm.as_ref().unwrap().forward(&q_2d)?;
                let k_2d = self.k_norm.as_ref().unwrap().forward(&k_2d)?;
                (
                    q_2d.reshape((seq_len, self.num_heads, self.head_dim))?,
                    k_2d.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
                )
            } else {
                // Per-head RMSNorm: flattening (S,H,D) -> (S*H,D) is equivalent to
                // applying norm directly on the 3D tensor when weight size is head_dim.
                let q_flat = q.flatten(0, 1)?;
                let k_flat = k.flatten(0, 1)?;
                let q_flat = self.q_norm.as_ref().unwrap().forward(&q_flat)?;
                let k_flat = self.k_norm.as_ref().unwrap().forward(&k_flat)?;
                (
                    q_flat.reshape((seq_len, self.num_heads, self.head_dim))?,
                    k_flat.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
                )
            }
        } else {
            (q, k)
        };

        Ok((q, k))
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_ext(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            None,
        )
    }

    pub fn forward_ext(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        q_scale: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_ext_with_probe(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            q_scale,
            None,
        )
    }

    /// Gemma 4 shared-KV attention. The preceding source layer has already
    /// written this call's keys and values into `source_cache`; this layer
    /// projects only Q and reads that cache without appending to it.
    pub fn forward_with_shared_kv(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        positions: &Tensor,
        source_cache: (&Tensor, &Tensor),
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        #[cfg(not(feature = "flash"))]
        candle_core::bail!("Gemma 4 shared attention requires the flash feature");

        #[cfg(feature = "flash")]
        {
            let (seq_len, _) = xs.dims2()?;
            let q_proj = match &self.qkv_proj {
                QkvProjection::QueryOnly(q_proj) => q_proj,
                _ => {
                    candle_core::bail!("Gemma 4 shared attention requires a query-only projection")
                }
            };
            if self.attn_output_gate {
                candle_core::bail!("Gemma 4 shared attention does not support output gating")
            }
            let q = q_proj
                .forward(xs)?
                .reshape((seq_len, self.num_heads, self.head_dim))?;
            let mut q =
                if let Some((q, _)) = self.fused_gemma4_qk_rope(&q, None, rotary_emb, positions)? {
                    q
                } else {
                    let mut q = q;
                    if self.promote_qk_to_f32 && q.dtype() != DType::F32 {
                        q = q.to_dtype(DType::F32)?;
                    }
                    if let Some(norm) = &self.q_norm {
                        q = if self.full_dim_qk_norm {
                            norm.forward(&q.reshape((seq_len, self.num_heads * self.head_dim))?)?
                                .reshape((seq_len, self.num_heads, self.head_dim))?
                        } else {
                            norm.forward(&q.flatten(0, 1)?)?.reshape((
                                seq_len,
                                self.num_heads,
                                self.head_dim,
                            ))?
                        };
                    }
                    if let Some(rope) = rotary_emb {
                        // The existing fused RoPE API rotates Q and K together.
                        let scratch_k =
                            Tensor::zeros((seq_len, 1, self.head_dim), q.dtype(), q.device())?;
                        if let Some((rotated_q, _)) =
                            rope.apply_rotary_emb_qkv(&q, &scratch_k, positions)?
                        {
                            q = rotated_q;
                        }
                    }
                    q
                };
            if self.qk_l2_norm {
                let rms = (q.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?;
                q = q.broadcast_div(&rms)?;
            }
            if q.dtype() != self.dtype {
                q = q.to_dtype(self.dtype)?;
            }
            let (key_cache, value_cache) = source_cache;
            if key_cache.dtype() != q.dtype() || value_cache.dtype() != q.dtype() {
                candle_core::bail!("Gemma 4 shared attention requires native K/V cache dtype")
            }
            let block_tables = input_metadata.block_tables.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("Gemma 4 shared attention requires block tables".into())
            })?;
            let context_lens = input_metadata.context_lens.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("Gemma 4 shared attention requires context lengths".into())
            })?;
            let context = if input_metadata.is_prefill {
                attention_rs::flash::flash_prefill(
                    &q,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.score_scale,
                    self.softcapping.unwrap_or(0.0) as f32,
                    self.sliding_window,
                    None,
                    None,
                    input_metadata.cu_seqlens_q.as_ref(),
                    input_metadata.max_seqlen_q,
                )?
            } else {
                let output = q.zeros_like()?;
                attention_rs::flash::flash_decode(
                    &q,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    &output,
                    input_metadata.max_context_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.score_scale,
                    self.softcapping.unwrap_or(0.0) as f32,
                    self.sliding_window,
                    None,
                    None,
                    self.shared_kv_splitk
                        .as_ref()
                        .and_then(|ws| ws.for_batch(seq_len)),
                )?
            };
            let y = context.reshape((seq_len, self.num_heads * self.head_dim))?;
            let y = if self.is_qvar_builder {
                y
            } else {
                y.to_dtype(xs.dtype())?
            };
            self.o_proj.forward(&y)
        }
    }

    /// Qwen3 experiment hook. The observer runs only when explicitly supplied.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_probe(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        layer_index: usize,
        observer: &AttentionProbeObserver,
    ) -> Result<Tensor> {
        self.forward_ext_with_probe(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            None,
            Some(AttentionHooks {
                layer_index,
                observer: Some(observer),
                sparse_provider: None,
            }),
        )
    }

    /// Qwen3 experiment hook for observation and optional exact-token sparse
    /// context replacement. Native attention still populates the KV cache.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_hooks(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        layer_index: usize,
        observer: Option<&AttentionProbeObserver>,
        sparse_provider: Option<&SparseAttentionProvider>,
    ) -> Result<Tensor> {
        self.forward_ext_with_probe(
            xs,
            rotary_emb,
            attention_mask,
            positions,
            cache,
            input_metadata,
            None,
            Some(AttentionHooks {
                layer_index,
                observer,
                sparse_provider,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_ext_with_probe(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        q_scale: Option<&Tensor>,
        hooks: Option<AttentionHooks<'_>>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (q_raw, k, v, head_views) = match &self.qkv_proj {
            QkvProjection::QueryOnly(_) => {
                candle_core::bail!("query-only attention requires forward_with_shared_kv")
            }
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
                false,
            ),
            QkvProjection::Packed(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                if qkv.len() != 3 {
                    candle_core::bail!(
                        "Expected 3 outputs from packed qkv projection, got {}",
                        qkv.len()
                    );
                }
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone(), false)
            }
            QkvProjection::PackedViews(qkv_proj) => {
                let qkv = qkv_proj.forward_head_views(xs, self.head_dim)?;
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone(), true)
            }
        };

        let local_q_dim = self.num_heads * self.head_dim;
        if head_views && self.attn_output_gate {
            candle_core::bail!("packed head views do not support a gated query projection")
        }
        let (q_linear, gate) = if self.attn_output_gate {
            let q_dim = q_raw.dim(1)?;
            if q_dim != local_q_dim * 2 {
                candle_core::bail!(
                    "q_proj output dim mismatch for gated attention, expected {}, got {}",
                    local_q_dim * 2,
                    q_dim
                );
            }
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2, 0, self.head_dim)?;
            let gate = q_gate.narrow(2, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let (q, k, v) = if head_views {
            // The fused Q/K transform accepts the packed row stride. The
            // paged KV writer expects V rows to be contiguous.
            (q_linear, k, v.contiguous()?)
        } else {
            (
                q_linear.reshape((seq_len, self.num_heads, self.head_dim))?,
                k.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
                v.reshape((seq_len, self.num_kv_heads, self.head_dim))?,
            )
        };

        let (q, k) = if let Some((q, Some(k))) =
            self.fused_gemma4_qk_rope(&q, Some(&k), rotary_emb, positions)?
        {
            (q, k)
        } else {
            let (q, k) = self.prepare_qk_for_rope(q, k, seq_len)?;
            if let Some(rotary_emb) = rotary_emb {
                match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
                    Some((q_new, k_new)) => (q_new, k_new),
                    None => (q, k),
                }
            } else {
                (q, k)
            }
        };

        let (q, k) = if self.qk_l2_norm {
            let q_rms = (q.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?;
            let k_rms = (k.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?;
            let q = q.broadcast_div(&q_rms)?;
            let k = k.broadcast_div(&k_rms)?;
            (q, k)
        } else {
            (q, k)
        };

        let (mut q, k) = if q.dtype() != self.dtype {
            let q = q.to_dtype(self.dtype)?;
            let k = k.to_dtype(self.dtype)?;
            (q, k)
        } else {
            (q, k)
        };

        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v
        };

        let v = if let Some(eps) = self.v_norm_eps {
            let orig_dtype = v.dtype();
            let v_f32 = v.to_dtype(DType::F32)?;
            let mean_sq = v_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
            let rms = (mean_sq + eps)?.sqrt()?;
            v_f32.broadcast_div(&rms)?.to_dtype(orig_dtype)?
        } else {
            v
        };

        if let Some(rotary_emb) = &rotary_emb {
            if let (Some(original_max_position_embeddings), Some(llama_4_scaling_beta)) = (
                rotary_emb.get_original_max_position_embeddings(),
                rotary_emb.get_llama_4_scaling_beta(),
            ) {
                use crate::utils::get_llama4_attn_scale;
                let scale = get_llama4_attn_scale(
                    &positions,
                    llama_4_scaling_beta,
                    original_max_position_embeddings as f64,
                )?
                .to_dtype(q.dtype())?;
                let scale = scale.squeeze(0)?.squeeze(0)?.reshape((seq_len, 1, 1))?;
                q = q.broadcast_mul(&scale)?;
            }
        }

        if let Some(scale) = q_scale {
            let scale = scale.reshape((seq_len, 1, 1))?;
            q = q
                .to_dtype(DType::F32)?
                .broadcast_mul(&scale.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
        }

        #[cfg(feature = "nemotron-attention-gemm")]
        let specialized = if self.nemotron_attention_gemm
            && std::env::var_os("XINFER_NEMOTRON_ATTENTION_GEMM").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            && self.sliding_window.is_none()
            && self.softcapping.is_none()
            && input_metadata.max_context_len >= 4096
            && q.dim(0)? >= 1024
        {
            crate::models::nemotron_h::attention_gemm::try_prefill(
                &q,
                &k,
                &v,
                cache,
                input_metadata,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.score_scale,
                std::env::var_os("XINFER_NEMOTRON_ATTENTION_GQA_WARPS").as_deref()
                    == Some(std::ffi::OsStr::new("1")),
            )?
        } else {
            None
        };
        #[cfg(not(feature = "nemotron-attention-gemm"))]
        let specialized: Option<Tensor> = None;
        let native_context = if let Some(context) = specialized {
            context
        } else {
            self.attn.forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                self.softcapping,
            )?
        };
        let mut context = native_context.clone();
        if let Some(hooks) = hooks {
            let mut probe = AttentionProbe {
                layer_index: hooks.layer_index,
                positions: positions.clone(),
                sequence_ids: input_metadata.sequence_ids.clone(),
                seqlens: input_metadata.seqlens.clone(),
                is_prefill: input_metadata.is_prefill,
                query: q.clone(),
                context: native_context.reshape((seq_len, self.num_heads, self.head_dim))?,
                effective_context: None,
                query_heads: self.num_heads,
                kv_heads: self.num_kv_heads,
                head_width: self.head_dim,
                dtype: q.dtype(),
                score_scale: self.score_scale,
            };
            if let Some(provider) = hooks.sparse_provider {
                let heads = provider(&probe)?;
                if let Some(replaced) = apply_sparse_attention_heads(
                    &probe,
                    heads,
                    self.sliding_window,
                    self.softcapping,
                )? {
                    context = replaced.clone();
                    probe.effective_context = Some(replaced);
                }
            }
            if let Some(observer) = hooks.observer {
                observer(probe)?;
            }
        }
        let y = context.reshape((seq_len, ()))?;

        let y = if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            y.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?
        } else {
            y
        };

        let y = if self.is_qvar_builder {
            y
        } else {
            y.to_dtype(xs.dtype())?
        };
        self.o_proj.forward(&y)
    }

    /// Optimized single-token attention without KV cache or paged attention.
    /// For seq_len=1, self-attention is trivially: softmax([1x1]) * V = V.
    /// We still compute Q/K/V projections and apply RoPE for correctness of
    /// the output projection, but skip the attention kernel entirely.
    pub fn forward_single_token_no_cache(
        &self,
        xs: &Tensor,
        rotary_emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let (q_raw, k, v) = match &self.qkv_proj {
            QkvProjection::QueryOnly(_) => {
                candle_core::bail!("query-only attention requires forward_with_shared_kv")
            }
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.forward(xs)?,
                k_proj.forward(xs)?,
                v_proj.forward(xs)?,
            ),
            QkvProjection::Packed(qkv_proj) | QkvProjection::PackedViews(qkv_proj) => {
                let qkv = qkv_proj.forward(xs)?;
                (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
            }
        };

        // Handle attn_output_gate (Qwen3.5 uses gated attention: Q proj outputs Q + gate)
        let local_q_dim = self.num_heads * self.head_dim;
        let (q_linear, gate) = if self.attn_output_gate {
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2, 0, self.head_dim)?;
            let gate = q_gate.narrow(2, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let q = q_linear.reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.prepare_qk_for_rope(q, k, seq_len)?;

        // Apply rotary embeddings (needed even for single token for positional encoding)
        let (_q, _k) = match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
            Some((q_new, k_new)) => (q_new, k_new),
            None => (q, k),
        };

        // For seq_len=1: softmax(Q*K^T / sqrt(d)) * V = V (single element softmax = 1.0)
        // Output shape: (seq_len, num_heads * head_dim) after GQA expansion
        let n_rep = self.num_heads / self.num_kv_heads;
        let y = if n_rep > 1 {
            v.unsqueeze(2)?
                .expand((seq_len, self.num_kv_heads, n_rep, self.head_dim))?
                .reshape((seq_len, self.num_heads * self.head_dim))?
        } else {
            v.reshape((seq_len, self.num_heads * self.head_dim))?
        };

        // Apply gated attention if needed
        let y = if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            y.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?
        } else {
            y
        };

        let y = if self.is_qvar_builder {
            y
        } else {
            y.to_dtype(xs.dtype())?
        };
        self.o_proj.forward(&y)
    }
}

#[cfg(test)]
mod packed_nvfp4_qkv_tests {
    use super::*;

    fn projection(input_scale: f32) -> Result<TensorParallelColumnLinear> {
        let device = candle_core::Device::Cpu;
        Ok(TensorParallelColumnLinear::new(LinearX::LnNvfp4(LnNvfp4 {
            blocks: Tensor::zeros((32, 16), DType::U8, &device)?,
            scales: Tensor::zeros((32, 2), DType::U8, &device)?,
            global_scale: 0.25,
            input_scale,
            bias: None,
            weight_scale_swizzled: None,
            resident_scales: None,
        })))
    }

    #[test]
    fn packing_requires_identical_calibration() -> Result<()> {
        let q = projection(0.5)?;
        let k = projection(0.5)?;
        let v = projection(0.25)?;
        assert!(Attention::try_pack_gemma4_nvfp4_qkv(&q, &k, &v)?.is_none());
        let v = projection(0.5)?;
        assert!(Attention::try_pack_gemma4_nvfp4_qkv(&q, &k, &v)?.is_some());
        Ok(())
    }
}

#[cfg(all(test, feature = "flash"))]
mod shared_kv_splitk_tests {
    use super::*;

    #[test]
    fn workspace_has_a_fixed_batch_bound() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let ws = SharedKvSplitKWorkspace::new(8, 8, 512, &device)?;
        assert_eq!(ws.tensor.elem_count(), 8 * 8 * 16 * 514);
        assert!(ws.for_batch(0).is_none());
        assert!(ws.for_batch(1).is_some());
        assert!(ws.for_batch(8).is_some());
        assert!(ws.for_batch(9).is_none());
        for (batch, heads, width) in [
            (0, 8, 512),
            (65, 8, 512),
            (8, 0, 512),
            (8, 8, 0),
            (8, usize::MAX, 512),
        ] {
            assert!(SharedKvSplitKWorkspace::new(batch, heads, width, &device).is_err());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn shared_kv_splitk_matches_cpu_softmax() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let (batch, heads, kv_heads, width, block, length) = (2, 8, 2, 512, 16, 8193usize);
        let pages = length.div_ceil(block);
        let mut rng = 0x1785_3624u32;
        let mut random = |len| {
            (0..len)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    (rng % 2001) as f32 / 1000. - 1.
                })
                .collect::<Vec<_>>()
        };
        let query = Tensor::from_vec(
            random(batch * heads * width),
            (batch, heads, width),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let key = Tensor::from_vec(
            random(pages * block * kv_heads * width),
            (pages, block, kv_heads, width),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let value = Tensor::from_vec(random(key.elem_count()), key.shape(), &device)?
            .to_dtype(DType::BF16)?;
        let table = (0..batch)
            .flat_map(|_| (0..pages).rev().map(|i| i as u32))
            .collect::<Vec<_>>();
        let table_gpu = Tensor::from_vec(table.clone(), (batch, pages), &device)?;
        let lengths = [length, 17];
        let lengths_gpu = Tensor::from_vec(lengths.map(|v| v as u32).to_vec(), batch, &device)?;
        let ws = SharedKvSplitKWorkspace::new(batch, heads, width, &device)?;
        let scale = 1. / (width as f32).sqrt();
        let mut outputs = Vec::new();
        for workspace in [None, ws.for_batch(batch)] {
            let output = Tensor::zeros(query.shape(), DType::BF16, &device)?;
            outputs.push(
                attention_rs::flash::flash_decode(
                    &query,
                    &key,
                    &value,
                    &table_gpu,
                    &lengths_gpu,
                    &output,
                    length,
                    heads,
                    kv_heads,
                    width,
                    scale,
                    0.,
                    None,
                    None,
                    None,
                    workspace,
                )?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            );
        }
        let q = query
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let k = key.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v = value
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut maximum = [0f64; 2];
        for (seq, &len) in lengths.iter().enumerate() {
            for head in 0..heads {
                let qb = (seq * heads + head) * width;
                let kv_base = |token: usize| {
                    let physical = table[seq * pages + token / block] as usize;
                    ((physical * block + token % block) * kv_heads + head / (heads / kv_heads))
                        * width
                };
                let scores = (0..len)
                    .map(|token| {
                        let kb = kv_base(token);
                        (0..width)
                            .map(|d| q[qb + d] as f64 * k[kb + d] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let denominator: f64 = p.iter().sum();
                for d in 0..width {
                    let expected = p
                        .iter()
                        .enumerate()
                        .map(|(t, p)| p * v[kv_base(t) + d] as f64)
                        .sum::<f64>()
                        / denominator;
                    for (i, output) in outputs.iter().enumerate() {
                        let actual = output[qb + d] as f64;
                        assert!(actual.is_finite());
                        maximum[i] = maximum[i].max((actual - expected).abs());
                    }
                }
            }
        }
        eprintln!(
            "shared_kv_splitk_cpu_oracle single_max_abs={} splitk_max_abs={}",
            maximum[0], maximum[1]
        );
        assert!(maximum.iter().all(|&delta| delta < 0.003));
        Ok(())
    }
}

pub struct NaiveAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
    softcapping: Option<f64>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilderX,
        num_heads: usize,
        hidden_size: usize,
        head_dim: usize,
        softcapping: Option<f64>,
        dtype: DType,
        key_mappings: HashMap<String, String>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let q_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["q_proj"])
            } else {
                vb.pp("q_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let k_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["k_proj"])
            } else {
                vb.pp("k_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let v_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["v_proj"])
            } else {
                vb.pp("v_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let o_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                if key_mappings.contains_key("o_proj") {
                    vb.pp(&key_mappings["o_proj"])
                } else {
                    vb.pp("o_proj")
                }
            },
            &None,
            &None,
            dtype,
        )?;

        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
            softcapping,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let shape = (b, seq_len, self.num_heads, self.head_dim);
        let q_packed = q
            .reshape(((), self.num_heads, self.head_dim))?
            .contiguous()?;
        let k_packed = k
            .reshape(((), self.num_heads, self.head_dim))?
            .contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let (q_packed, k_packed) = if let Some(positions) = positions {
            match emb.apply_rotary_emb_qkv(&q_packed, &k_packed, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q_packed, k_packed), // In-place operation, keep originals
            }
        } else {
            (q_packed, k_packed)
        };

        let q = q_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k_packed
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let chunk_size = 1024;
        let mut attn_chunks = Vec::new();
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        for c in 0..num_chunks {
            let offset = c * chunk_size;
            let len = chunk_size.min(seq_len - offset);
            //chunk at query is correct for the following
            let q_chunk = q.narrow(2, offset, len)?.contiguous()?;
            let mut att = (q_chunk.matmul(&k.t()?)? * f64::from(self.scale))?;

            if let Some(sc) = self.softcapping {
                att = ((att / sc)?.tanh()? * sc)?;
            }
            if let Some(mask) = &mask {
                //mask needs to be chunked
                let q_chunk_mask = mask.narrow(2, offset, len)?; // shape: [1, 1, chunk_len, K_len]
                att = att.broadcast_add(&q_chunk_mask)?;
            }
            att = candle_nn::ops::softmax_last_dim(&att.to_dtype(candle_core::DType::F32)?)?
                .to_dtype(att.dtype())?;

            let att_chunk = att.matmul(&v)?;
            attn_chunks.push(att_chunk);
        }

        let att = Tensor::cat(&attn_chunks, 2)?
            .contiguous()?
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((b, seq_len, ()))?;
        self.o_proj.forward(&att)
    }
}
