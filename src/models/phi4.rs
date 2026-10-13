// src/models/phi4.rs
// This implementation is adapted from Phi-3, using the local Candle layers.
use crate::models::layers::distributed::{
    kv_head_shard, shard, Comm, MergedParallelColumnLinear, TensorParallelRowLinear,
    VocabParallelLinear,
};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, RotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::{Config, RopeScalingValue};
use crate::utils::progress::ProgressLike;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

/// Persist this alongside a Phi KV checkpoint. It describes how its keys were
/// rotated; a restored cache must be bound before continued inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phi4RopeCacheBinding {
    pub version: u32,
    pub prefix_len: usize,
    pub uses_long_rope: bool,
    pub rope_fingerprint: [u8; 32],
}

#[derive(Debug, Clone)]
struct Phi4RotaryEmbedding {
    normal_emb: RotaryEmbedding,
    long_emb: Option<RotaryEmbedding>,
    original_max_position_embeddings: Option<usize>,
    fingerprint: [u8; 32],
}

impl Phi4RotaryEmbedding {
    fn original_max_seq_len(cfg: &Config) -> Result<usize> {
        let nested = cfg
            .rope_scaling
            .as_ref()
            .and_then(|rope| rope.get("original_max_position_embeddings"));
        let original = match nested {
            Some(RopeScalingValue::Number(value))
                if value.is_finite()
                    && *value >= 2.0
                    && value.fract() == 0.0
                    && *value <= usize::MAX as f64 =>
            {
                *value as usize
            }
            Some(_) => candle_core::bail!(
                "rope_scaling.original_max_position_embeddings must be an integer >= 2"
            ),
            None => cfg
                .original_max_position_embeddings
                .unwrap_or(cfg.max_position_embeddings),
        };
        if original < 2 || original > cfg.max_position_embeddings {
            candle_core::bail!(
                "original_max_position_embeddings must be in 2..=max_position_embeddings"
            );
        }
        Ok(original)
    }

    fn effective_max_seq_len(cfg: &Config) -> Result<usize> {
        let Some(rope_scaling) = &cfg.rope_scaling else {
            return Ok(cfg.max_position_embeddings);
        };

        let factor = match rope_scaling.get("factor") {
            None => None,
            Some(value) => match value.as_f64() {
                Some(factor) if factor.is_finite() && factor >= 1.0 => Some(factor),
                _ => candle_core::bail!("Phi rope_scaling.factor must be finite and >= 1"),
            },
        };
        let original_max_position_embeddings = Self::original_max_seq_len(cfg)? as f64;

        Ok(match factor {
            Some(factor) if factor > 1.0 => std::cmp::max(
                cfg.max_position_embeddings,
                (original_max_position_embeddings * factor).round() as usize,
            ),
            _ => cfg.max_position_embeddings,
        })
    }

    fn rope_scaling_array(
        value: &RopeScalingValue,
        expected_len: usize,
        name: &str,
    ) -> Result<Vec<f64>> {
        match value {
            RopeScalingValue::NumberArray(v) => {
                if v.len() != expected_len {
                    candle_core::bail!(
                        "{name} length mismatch: expected {expected_len}, got {}",
                        v.len()
                    );
                }
                if v.iter().any(|factor| !factor.is_finite() || *factor <= 0.0) {
                    candle_core::bail!("{name} factors must be finite and positive");
                }
                Ok(v.clone())
            }
            RopeScalingValue::Number(v) if v.is_finite() && *v > 0.0 => Ok(vec![*v; expected_len]),
            _ => candle_core::bail!("{name} must be a number array"),
        }
    }

    fn new(dtype: DType, cfg: &Config, is_rope_i: bool, dev: &Device) -> Result<Self> {
        if cfg
            .partial_rotary_factor
            .is_some_and(|factor| !factor.is_finite() || factor <= 0.0 || factor > 1.0)
        {
            candle_core::bail!("Phi partial_rotary_factor must be finite and in (0, 1]");
        }
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            candle_core::bail!("Phi rotary dimension must be positive and even");
        }
        let max_seq_len = Self::effective_max_seq_len(cfg)?;
        if max_seq_len > u32::MAX as usize {
            candle_core::bail!("Phi maximum position exceeds rotary table index range");
        }
        let rope_theta = cfg.rope_theta.unwrap_or(10000.0);
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            candle_core::bail!("Phi rope_theta must be finite and positive");
        }
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        if let Some(rope_scaling) = &cfg.rope_scaling {
            let original_max_position_embeddings = Self::original_max_seq_len(cfg)?;
            let rope_type = rope_scaling
                .get("type")
                .or_else(|| rope_scaling.get("rope_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("default");

            let short_factor = rope_scaling
                .get("short_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing short_factor"))?;
            let long_factor = rope_scaling
                .get("long_factor")
                .ok_or_else(|| candle_core::Error::msg("rope_scaling missing long_factor"))?;
            let short_factor =
                Self::rope_scaling_array(short_factor, inv_freq_len, "short_factor")?;
            let long_factor = Self::rope_scaling_array(long_factor, inv_freq_len, "long_factor")?;

            // Compute scaling factor (same for both short and long embeddings)
            let scale = max_seq_len as f64 / original_max_position_embeddings as f64;
            let scaling_factor = if scale <= 1.0 {
                1.0
            } else {
                match rope_type {
                    "su" | "longrope" => {
                        (1.0 + scale.ln() / (original_max_position_embeddings as f64).ln()).sqrt()
                    }
                    "yarn" => 0.1 * scale.ln() + 1.0,
                    _ => 1.0,
                }
            };

            let inv_freq_long = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (long_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64))) as f32
                })
                .collect::<Vec<_>>();
            let inv_freq_short = (0..rotary_dim)
                .step_by(2)
                .enumerate()
                .map(|(k, i)| {
                    (1f64 / (short_factor[k] * rope_theta.powf(i as f64 / rotary_dim as f64)))
                        as f32
                })
                .collect::<Vec<_>>();

            let inv_freq_long =
                Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
            let inv_freq_short =
                Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;

            let freqs_long = t.matmul(&inv_freq_long)?;
            let long_sin = (freqs_long.sin()? * scaling_factor)?;
            let long_cos = (freqs_long.cos()? * scaling_factor)?;

            let freqs_short = t.matmul(&inv_freq_short)?;
            let short_sin = (freqs_short.sin()? * scaling_factor)?;
            let short_cos = (freqs_short.cos()? * scaling_factor)?;

            let normal_emb = RotaryEmbedding {
                sin: short_sin.to_dtype(dtype)?,
                cos: short_cos.to_dtype(dtype)?,
                is_rope_i,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
                original_max_position_embeddings: Some(original_max_position_embeddings),
                llama_4_scaling_beta: None,
            };

            let long_emb = RotaryEmbedding {
                sin: long_sin.to_dtype(dtype)?,
                cos: long_cos.to_dtype(dtype)?,
                is_rope_i,
                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                    Some(rotary_dim)
                } else {
                    None
                },
                original_max_position_embeddings: Some(original_max_position_embeddings),
                llama_4_scaling_beta: None,
            };

            return Ok(Self {
                normal_emb,
                long_emb: Some(long_emb),
                original_max_position_embeddings: Some(original_max_position_embeddings),
                fingerprint: Self::fingerprint(
                    cfg,
                    original_max_position_embeddings,
                    max_seq_len,
                    rotary_dim,
                    &short_factor,
                    &long_factor,
                    (dtype, is_rope_i),
                ),
            });
        }

        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs = t.matmul(&inv_freq)?;

        let normal_emb = RotaryEmbedding {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            is_rope_i,
            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                Some(rotary_dim)
            } else {
                None
            },
            original_max_position_embeddings: None,
            llama_4_scaling_beta: None,
        };
        Ok(Self {
            normal_emb,
            long_emb: None,
            original_max_position_embeddings: None,
            fingerprint: Self::fingerprint(
                cfg,
                max_seq_len,
                max_seq_len,
                rotary_dim,
                &[],
                &[],
                (dtype, is_rope_i),
            ),
        })
    }

    fn fingerprint(
        cfg: &Config,
        original: usize,
        max_seq_len: usize,
        rotary_dim: usize,
        short: &[f64],
        long: &[f64],
        kernel_layout: (DType, bool),
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"xinfer-phi4-rope-v1");
        for value in [
            cfg.max_position_embeddings,
            original,
            max_seq_len,
            rotary_dim,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim
                .unwrap_or(cfg.hidden_size / cfg.num_attention_heads),
        ] {
            hasher.update((value as u64).to_le_bytes());
        }
        hasher.update(cfg.rope_theta.unwrap_or(10000.0).to_bits().to_le_bytes());
        hasher.update(format!("{:?}", kernel_layout.0).as_bytes());
        hasher.update([kernel_layout.1 as u8]);
        if let Some(rope) = &cfg.rope_scaling {
            let rope_type = rope
                .get("type")
                .or_else(|| rope.get("rope_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("default");
            hasher.update(rope_type.as_bytes());
        }
        for factors in [short, long] {
            hasher.update((factors.len() as u64).to_le_bytes());
            for value in factors {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        hasher.finalize().into()
    }

    fn uses_long_rope(&self, end_position: usize) -> bool {
        self.original_max_position_embeddings
            .is_some_and(|original| end_position > original)
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
        use_long_rope: bool,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if use_long_rope {
            self.long_emb
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("Phi long RoPE is unavailable"))?
                .apply_rotary_emb_qkv(q, k, input_positions)
        } else {
            self.normal_emb.apply_rotary_emb_qkv(q, k, input_positions)
        }
    }
}

struct Phi4Attention {
    qkv_proj: MergedParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Phi4RotaryEmbedding>,
    attn: PagedAttention,
    is_quantized: bool,
    dtype: DType,
}

impl Phi4Attention {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let is_qvar_builder = vb.is_qvar_builder();

        let world_size = comm.world_size();
        let attention_heads = num_heads / world_size;
        let (kv_heads, kv_shard) = kv_head_shard(num_kv_heads, comm.rank(), world_size)?;
        let kv_shard_rank = kv_shard.rank;
        let kv_shard_world_size = kv_shard.world_size;

        let qkv_proj = MergedParallelColumnLinear::load_merged_chunks(
            cfg.hidden_size,
            op_size,
            0,
            vec![
                num_heads * head_dim,
                num_kv_heads * head_dim,
                num_kv_heads * head_dim,
            ],
            Some(vec![
                shard(0, comm.rank(), world_size),
                shard(0, kv_shard_rank, kv_shard_world_size),
                shard(0, kv_shard_rank, kv_shard_world_size),
            ]),
            if is_qvar_builder {
                vb.pp("attn_qkv")
            } else {
                vb.pp("qkv_proj")
            },
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;
        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
            cfg.hidden_size,
            if is_qvar_builder {
                vb.pp("attn_output")
            } else {
                vb.pp("o_proj")
            },
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(kv_heads),
                cfg.sliding_window,
                vb.device().clone(),
                None,
                cfg.kvcache_dtype.is_fp8_keys(),
            )?,
            is_quantized: is_qvar_builder || cfg.quant.is_some(),
            dtype,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        use_long_rope: bool,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv[0].reshape((seq_len, self.num_heads, self.head_dim))?;
        let k = qkv[1].reshape((seq_len, self.num_kv_heads, self.head_dim))?;
        let v = qkv[2].reshape((seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) =
            match self
                .rotary_emb
                .apply_rotary_emb_qkv(&q, &k, input_positions, use_long_rope)?
            {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q, k), // In-place operation, keep originals
            };

        let (q, k, v) = if self.is_quantized {
            (
                q.to_dtype(self.dtype)?,
                k.to_dtype(self.dtype)?,
                v.to_dtype(self.dtype)?,
            )
        } else {
            (q, k, v)
        };

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((seq_len, ()))?;

        self.o_proj.forward(&y)?.to_dtype(xs.dtype())
    }
}

struct Phi4DecoderLayer {
    self_attn: Phi4Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Phi4DecoderLayer {
    fn new(
        rotary_emb: Arc<Phi4RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let self_attn = Phi4Attention::new(
            rotary_emb,
            cfg,
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn")
            },
            comm.clone(),
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp")
            },
            comm.clone(),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.hidden_act,
            &cfg.quantization_config,
            &cfg.quant,
            true,
            dtype,
            "",
        )?;
        let key_map: HashMap<&str, &str> = [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
        .iter()
        .cloned()
        .collect();
        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["input_layernorm"])
            } else {
                vb.pp("input_layernorm")
            },
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["post_attention_layernorm"])
            } else {
                vb.pp("post_attention_layernorm")
            },
            DType::F32,
            false,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        use_long_rope: bool,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            input_positions,
            use_long_rope,
            cache,
            input_metadata,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

pub struct Phi4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Phi4DecoderLayer>,
    norm: NormX,
    lm_head: VocabParallelLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
    rotary_emb: Arc<Phi4RotaryEmbedding>,
    rope_cache_bindings: RwLock<HashMap<usize, Phi4RopeCacheBinding>>,
}

impl Phi4ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("model.embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();
        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["model.embed_tokens"])
            } else {
                vb.pp("model.embed_tokens")
            },
            dtype,
        )?;
        let rotary_emb = Arc::new(Phi4RotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            config,
            is_rope_i,
            &vb.device(),
        )?);
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = Phi4DecoderLayer::new(
                rotary_emb.clone(),
                config,
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        key_map["model.layers"]
                    } else {
                        "model.layers"
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }
        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["model.norm"])
            } else {
                vb.pp("model.norm")
            },
            DType::F32,
            false,
        )?;
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(key_map["model.embed_tokens"])
                } else {
                    vb.pp("model.embed_tokens")
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("lm_head")
                }
            },
            comm.clone(),
            &config.quantization_config,
            &None,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
            rotary_emb,
            rope_cache_bindings: RwLock::new(HashMap::new()),
        })
    }

    /// Return the position settings to persist with an existing sequence's KV.
    pub fn export_rope_cache_binding(&self, seq_id: usize) -> Result<Phi4RopeCacheBinding> {
        self.rope_cache_bindings
            .read()
            .get(&seq_id)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Phi sequence has no RoPE cache binding"))
    }

    /// Attach restored KV to a sequence only when its saved RoPE layout matches.
    /// The caller must restore the matching KV through its normal cache path first.
    pub fn bind_restored_rope_cache(
        &self,
        seq_id: usize,
        binding: &Phi4RopeCacheBinding,
    ) -> Result<()> {
        if binding.version != 1
            || binding.prefix_len == 0
            || binding.prefix_len > Self::max_rope_position(&self.rotary_emb)
            || binding.rope_fingerprint != self.rotary_emb.fingerprint
        {
            candle_core::bail!("incompatible Phi RoPE cache binding");
        }
        if let Some(original) = self.rotary_emb.original_max_position_embeddings {
            if binding.prefix_len > original && !binding.uses_long_rope {
                candle_core::bail!("Phi cache past the original context cannot use short RoPE");
            }
        } else if binding.uses_long_rope {
            candle_core::bail!("Phi cache binding requests unavailable long RoPE");
        }
        let mut bindings = self.rope_cache_bindings.write();
        if bindings.contains_key(&seq_id) {
            candle_core::bail!("Phi sequence already has a RoPE cache binding");
        }
        bindings.insert(seq_id, binding.clone());
        Ok(())
    }

    pub fn release_sequence_state(&self, seq_id: usize) {
        self.rope_cache_bindings.write().remove(&seq_id);
    }

    /// Select long RoPE before the first chunk of a known long prompt. This
    /// lets a 4K boundary be crossed in multiple prefill calls with coherent KV.
    pub fn prepare_long_rope_cache(&self, seq_id: usize, planned_total_len: usize) -> Result<()> {
        if !self.rotary_emb.uses_long_rope(planned_total_len)
            || planned_total_len > Self::max_rope_position(&self.rotary_emb)
        {
            candle_core::bail!("planned Phi prompt does not require long RoPE");
        }
        let mut bindings = self.rope_cache_bindings.write();
        if bindings.contains_key(&seq_id) {
            candle_core::bail!("Phi sequence already has a RoPE cache binding");
        }
        bindings.insert(
            seq_id,
            Phi4RopeCacheBinding {
                version: 1,
                prefix_len: 0,
                uses_long_rope: true,
                rope_fingerprint: self.rotary_emb.fingerprint,
            },
        );
        Ok(())
    }

    fn max_rope_position(rotary: &Phi4RotaryEmbedding) -> usize {
        rotary.normal_emb.sin.dims()[0]
    }

    fn check_rope_cache(
        &self,
        positions: &Tensor,
        metadata: &InputMetadata,
    ) -> Result<(bool, Vec<(usize, Phi4RopeCacheBinding)>)> {
        let positions = positions.flatten_all()?.to_vec1::<i64>()?;
        let mut updates = Vec::new();
        let mut start = 0usize;
        let boundaries: Vec<usize> = if let Some(seqlens) = &metadata.seqlens {
            seqlens.iter().map(|&n| n as usize).collect()
        } else {
            (1..=positions.len()).collect()
        };
        if boundaries.last().copied().unwrap_or(0) != positions.len() {
            candle_core::bail!("Phi position/sequence metadata length mismatch");
        }
        let ids = metadata.sequence_ids.as_ref();
        if ids.is_some_and(|ids| ids.len() != boundaries.len()) {
            candle_core::bail!("Phi sequence id count mismatch");
        }
        let bindings = self.rope_cache_bindings.read();
        let mut batch_regime = None;
        for (index, end) in boundaries.into_iter().enumerate() {
            if end <= start {
                candle_core::bail!("Phi empty or unordered sequence span");
            }
            let segment = &positions[start..end];
            let first = segment[0];
            if first < 0
                || segment
                    .iter()
                    .enumerate()
                    .any(|(offset, &pos)| pos != first + offset as i64)
            {
                candle_core::bail!(
                    "Phi positions must be nonnegative and contiguous within each sequence"
                );
            }
            let next_len = first as usize + segment.len();
            if next_len > Self::max_rope_position(&self.rotary_emb) {
                candle_core::bail!("Phi position exceeds configured rotary table");
            }
            let id = ids.map(|ids| ids[index]);
            let saved = id.and_then(|id| bindings.get(&id));
            let long = saved
                .map(|binding| binding.uses_long_rope)
                .unwrap_or_else(|| self.rotary_emb.uses_long_rope(next_len));
            if !long && self.rotary_emb.uses_long_rope(next_len) {
                candle_core::bail!("Phi short RoPE cache crosses original context; rebuild or prepare long RoPE before prefill");
            }
            if batch_regime.is_some_and(|other| other != long) {
                candle_core::bail!(
                    "Phi batch mixes short and long RoPE sequences; split the batch"
                );
            }
            batch_regime = Some(long);
            if let Some(id) = id {
                if let Some(saved) = saved {
                    if saved.prefix_len != first as usize {
                        candle_core::bail!("Phi RoPE cache regime or prefix mismatch; rebuild the cache with one regime");
                    }
                } else if first != 0 {
                    candle_core::bail!("Phi continued or restored cache lacks a RoPE binding");
                }
                updates.push((
                    id,
                    Phi4RopeCacheBinding {
                        version: 1,
                        prefix_len: next_len,
                        uses_long_rope: long,
                        rope_fingerprint: self.rotary_emb.fingerprint,
                    },
                ));
            } else if first != 0 {
                candle_core::bail!(
                    "Phi continuation requires a sequence id and RoPE cache binding"
                );
            }
            start = end;
        }
        Ok((batch_regime.unwrap_or(false), updates))
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)
        } else {
            Ok(xs)
        }
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let (use_long_rope, rope_updates) = if kv_caches.is_some() {
            self.check_rope_cache(positions, input_metadata)?
        } else {
            let end = positions
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .max()
                .unwrap_or(0)
                + 1;
            (self.rotary_emb.uses_long_rope(end as usize), Vec::new())
        };
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    use_long_rope,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1_u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        let output = if return_hidden {
            xs.to_dtype(DType::F32)
        } else if self.is_qvar_builder {
            self.lm_head.forward(&xs)
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }?;
        if !rope_updates.is_empty() {
            self.rope_cache_bindings.write().extend(rope_updates);
        }
        Ok(output)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            true,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn phi_config() -> Config {
        let short = vec![1.0_f64; 48];
        let long = (0..48).map(|i| 1.0 + i as f64 / 8.0).collect::<Vec<_>>();
        serde_json::from_value(json!({
            "architectures": ["Phi3ForCausalLM"],
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "partial_rotary_factor": 0.75,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "longrope", "short_factor": short, "long_factor": long}
        }))
        .unwrap()
    }

    #[test]
    fn phi_root_original_context_and_nested_precedence() {
        let mut cfg = phi_config();
        assert_eq!(cfg.original_max_position_embeddings, Some(4096));
        assert_eq!(
            Phi4RotaryEmbedding::original_max_seq_len(&cfg).unwrap(),
            4096
        );
        assert_eq!(
            Phi4RotaryEmbedding::effective_max_seq_len(&cfg).unwrap(),
            131072
        );
        cfg.rope_scaling.as_mut().unwrap().insert(
            "original_max_position_embeddings".into(),
            RopeScalingValue::Number(8192.0),
        );
        assert_eq!(
            Phi4RotaryEmbedding::original_max_seq_len(&cfg).unwrap(),
            8192
        );
        cfg.rope_scaling.as_mut().unwrap().insert(
            "original_max_position_embeddings".into(),
            RopeScalingValue::Number(4096.5),
        );
        assert!(Phi4RotaryEmbedding::original_max_seq_len(&cfg).is_err());
        cfg.rope_scaling
            .as_mut()
            .unwrap()
            .remove("original_max_position_embeddings");
        cfg.original_max_position_embeddings = Some(0);
        assert!(Phi4RotaryEmbedding::original_max_seq_len(&cfg).is_err());
        cfg.original_max_position_embeddings = Some(4096);
        cfg.rope_scaling
            .as_mut()
            .unwrap()
            .insert("factor".into(), RopeScalingValue::Number(f64::NAN));
        assert!(Phi4RotaryEmbedding::effective_max_seq_len(&cfg).is_err());
        cfg.rope_scaling.as_mut().unwrap().remove("factor");
        cfg.partial_rotary_factor = Some(1.5);
        assert!(Phi4RotaryEmbedding::new(DType::F32, &cfg, false, &Device::Cpu).is_err());
    }

    #[test]
    fn phi_longrope_frequency_amplitude_and_boundary() {
        let cfg = phi_config();
        let emb = Phi4RotaryEmbedding::new(DType::F32, &cfg, false, &Device::Cpu).unwrap();
        assert_eq!(emb.original_max_position_embeddings, Some(4096));
        assert_eq!(emb.normal_emb.rotary_dim, Some(96));
        assert!(!emb.uses_long_rope(4096));
        assert!(emb.uses_long_rope(4097));
        let amplitude = (1.0_f64 + 32.0_f64.ln() / 4096.0_f64.ln()).sqrt();
        let short_cos = emb
            .normal_emb
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let long_cos = emb
            .long_emb
            .as_ref()
            .unwrap()
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!((short_cos[0][0] as f64 - amplitude).abs() < 1e-5);
        assert!((long_cos[0][0] as f64 - amplitude).abs() < 1e-5);
        let position = 4096usize;
        let channel = 1usize;
        let theta = 10000.0_f64;
        let short_frequency = 1.0 / theta.powf(2.0 / 96.0);
        let long_frequency = short_frequency / (1.0 + channel as f64 / 8.0);
        let actual_short = emb
            .normal_emb
            .sin
            .narrow(0, position, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][channel] as f64;
        let actual_long = emb
            .long_emb
            .as_ref()
            .unwrap()
            .sin
            .narrow(0, position, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][channel] as f64;
        // The table multiplies a single-precision frequency by a single-precision position.
        let expected_short = ((position as f32 * short_frequency as f32).sin() as f64) * amplitude;
        let expected_long = ((position as f32 * long_frequency as f32).sin() as f64) * amplitude;
        assert!((actual_short - expected_short).abs() < 2e-4);
        assert!((actual_long - expected_long).abs() < 2e-4);
        assert!((actual_short - actual_long).abs() > 0.01);
    }
}
