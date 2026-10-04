//! Nemotron-H: a sequence of Mamba2, MLP, and unrotated attention blocks.
//! The recurrent path is deliberately expressed in Candle operations so the
//! checkpoint can run without the optional mamba-ssm CUDA extension.
mod state;

pub use state::{
    NemotronMambaSnapshot, NemotronMambaStateLayout, NemotronStateDType,
    NEMOTRON_MAMBA_STATE_VERSION,
};

use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear, VocabParallelLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::moe::MoeRouting;
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Deserialize)]
struct NemotronConfig {
    hybrid_override_pattern: String,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    n_groups: usize,
    ssm_state_size: usize,
    conv_kernel: usize,
    mamba_hidden_act: String,
    mlp_hidden_act: String,
    #[serde(default)]
    use_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default = "default_eps")]
    layer_norm_epsilon: f64,
    #[serde(default)]
    mamba_dt_limit: Option<(f64, f64)>,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    moe_shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    routed_scaling_factor: Option<f64>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    n_group: Option<usize>,
    #[serde(default)]
    topk_group: Option<usize>,
}

fn default_eps() -> f64 {
    1e-5
}

impl NemotronConfig {
    fn from_config(config: &Config) -> Result<Self> {
        let raw = config.extra_config_json.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("Nemotron-H requires the original config.json".into())
        })?;
        let cfg: Self = serde_json::from_str(raw).map_err(candle_core::Error::wrap)?;
        if cfg.hybrid_override_pattern.len() != config.num_hidden_layers
            || !cfg
                .hybrid_override_pattern
                .bytes()
                .all(|b| matches!(b, b'M' | b'*' | b'-' | b'E'))
            || cfg.n_groups == 0
            || cfg.mamba_num_heads == 0
            || cfg.mamba_head_dim == 0
            || cfg.ssm_state_size == 0
            || cfg.mamba_num_heads % cfg.n_groups != 0
            || cfg.conv_kernel == 0
            || cfg.mamba_hidden_act != "silu"
            || cfg.mlp_hidden_act != "relu2"
            || cfg.use_bias
            || cfg.mlp_bias
        {
            candle_core::bail!("invalid Nemotron-H layer pattern or Mamba dimensions")
        }
        if cfg.hybrid_override_pattern.contains('E') {
            let experts = cfg.n_routed_experts.unwrap_or(0);
            let top_k = cfg.num_experts_per_tok.unwrap_or(0);
            if experts == 0
                || top_k == 0
                || top_k > experts
                || cfg.moe_intermediate_size.unwrap_or(0) == 0
                || cfg.moe_shared_expert_intermediate_size.unwrap_or(0) == 0
                || cfg.n_group.unwrap_or(0) == 0
                || cfg.topk_group.unwrap_or(0) == 0
            {
                candle_core::bail!("invalid Nemotron-H MoE dimensions")
            }
        }
        Ok(cfg)
    }
}

#[derive(Clone)]
struct MambaState {
    conv: Tensor, // [kernel - 1, conv_dim], oldest first
    ssm: Tensor,  // [heads, head_dim, state_dim], always F32
}

struct Mamba2 {
    in_proj: ReplicatedLinear,
    out_proj: ReplicatedLinear,
    conv_weight: Tensor,
    conv_bias: Tensor,
    dt_bias: Tensor,
    a: Tensor,
    d: Tensor,
    norm_weight: Tensor,
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_dim: usize,
    kernel: usize,
    conv_dim: usize,
    eps: f64,
    dt_limit: (f64, f64),
    device: Device,
    dtype: DType,
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros_like(x)?;
    let positive = x.broadcast_maximum(&zero)?;
    let negative_abs = x.broadcast_minimum(&x.neg()?)?;
    positive.broadcast_add(&(negative_abs.exp()? + 1.0)?.log()?)
}

impl Mamba2 {
    fn new(vb: VarBuilderX, config: &Config, c: &NemotronConfig, dtype: DType) -> Result<Self> {
        let inner = c.mamba_num_heads * c.mamba_head_dim;
        let conv_dim = inner + 2 * c.n_groups * c.ssm_state_size;
        let in_dim = inner + conv_dim + c.mamba_num_heads;
        let in_proj = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            in_dim,
            vb.pp("in_proj"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let out_proj = ReplicatedLinear::load_no_bias(
            inner,
            config.hidden_size,
            vb.pp("out_proj"),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let shard = candle_nn::var_builder::Shard::default();
        let conv_weight = vb
            .get_with_hints_dtype(
                (conv_dim, 1, c.conv_kernel),
                "conv1d.weight",
                shard,
                DType::F32,
            )?
            .reshape((conv_dim, c.conv_kernel))?
            .transpose(0, 1)?
            .contiguous()?;
        let conv_bias = vb.get_with_hints_dtype((conv_dim,), "conv1d.bias", shard, DType::F32)?;
        let dt_bias =
            vb.get_with_hints_dtype((c.mamba_num_heads,), "dt_bias", shard, DType::F32)?;
        let a = vb
            .get_with_hints_dtype((c.mamba_num_heads,), "A_log", shard, DType::F32)?
            .exp()?
            .neg()?;
        let d = vb.get_with_hints_dtype((c.mamba_num_heads,), "D", shard, DType::F32)?;
        let norm_weight = vb.get_with_hints_dtype((inner,), "norm.weight", shard, DType::F32)?;
        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            conv_bias,
            dt_bias,
            a,
            d,
            norm_weight,
            heads: c.mamba_num_heads,
            head_dim: c.mamba_head_dim,
            groups: c.n_groups,
            state_dim: c.ssm_state_size,
            kernel: c.conv_kernel,
            conv_dim,
            eps: c.layer_norm_epsilon,
            dt_limit: c.mamba_dt_limit.unwrap_or((0.0, f64::INFINITY)),
            device: vb.device().clone(),
            dtype,
        })
    }

    fn zero_state(&self) -> Result<MambaState> {
        Ok(MambaState {
            conv: Tensor::zeros((self.kernel - 1, self.conv_dim), DType::F32, &self.device)?,
            ssm: Tensor::zeros(
                (self.heads, self.head_dim, self.state_dim),
                DType::F32,
                &self.device,
            )?,
        })
    }

    fn step(&self, projected: &Tensor, state: &mut MambaState) -> Result<Tensor> {
        let inner = self.heads * self.head_dim;
        let gate = projected.narrow(0, 0, inner)?.to_dtype(DType::F32)?;
        let conv_in = projected
            .narrow(0, inner, self.conv_dim)?
            .to_dtype(DType::F32)?;
        let dt = projected
            .narrow(0, inner + self.conv_dim, self.heads)?
            .to_dtype(DType::F32)?;
        let window = Tensor::cat(&[&state.conv, &conv_in.unsqueeze(0)?], 0)?;
        let conv = window
            .broadcast_mul(&self.conv_weight)?
            .sum(0)?
            .broadcast_add(&self.conv_bias)?;
        let conv = candle_nn::ops::silu(&conv)?;
        state.conv = window.narrow(0, 1, self.kernel - 1)?.contiguous()?;

        let x = conv
            .narrow(0, 0, inner)?
            .reshape((self.heads, self.head_dim))?;
        let group_size = self.groups * self.state_dim;
        let repeat = self.heads / self.groups;
        let expand_groups = |offset| -> Result<Tensor> {
            conv.narrow(0, offset, group_size)?
                .reshape((self.groups, 1, self.state_dim))?
                .broadcast_as((self.groups, repeat, self.state_dim))?
                .contiguous()?
                .reshape((self.heads, self.state_dim))
        };
        let b = expand_groups(inner)?;
        let c = expand_groups(inner + group_size)?;
        let dt =
            softplus(&dt.broadcast_add(&self.dt_bias)?)?.clamp(self.dt_limit.0, self.dt_limit.1)?;
        let decay = dt
            .broadcast_mul(&self.a)?
            .exp()?
            .reshape((self.heads, 1, 1))?;
        let input = x
            .broadcast_mul(&dt.unsqueeze(1)?)?
            .unsqueeze(2)?
            .broadcast_mul(&b.unsqueeze(1)?)?;
        state.ssm = state.ssm.broadcast_mul(&decay)?.broadcast_add(&input)?;
        let y = state
            .ssm
            .broadcast_mul(&c.unsqueeze(1)?)?
            .sum(2)?
            .broadcast_add(&x.broadcast_mul(&self.d.unsqueeze(1)?)?)?;

        // MambaRMSNormGated with norm_before_gate=False and one norm per group.
        let gated = y
            .reshape((inner,))?
            .broadcast_mul(&candle_nn::ops::silu(&gate)?)?;
        let grouped = gated.reshape((self.groups, inner / self.groups))?;
        let variance = grouped.sqr()?.mean_keepdim(1)?;
        let normalized = grouped
            .broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .reshape((inner,))?
            .broadcast_mul(&self.norm_weight)?;
        self.out_proj
            .forward(&normalized.unsqueeze(0)?.to_dtype(self.dtype)?)
    }

    fn forward(
        &self,
        xs: &Tensor,
        spans: &[(usize, usize, usize)],
        states: &mut HashMap<usize, Vec<Option<MambaState>>>,
        layer: usize,
        num_layers: usize,
    ) -> Result<Tensor> {
        let projected = self.in_proj.forward(xs)?;
        let mut outputs = Vec::with_capacity(xs.dim(0)?);
        for &(seq_id, start, end) in spans {
            let entry = states
                .entry(seq_id)
                .or_insert_with(|| vec![None; num_layers]);
            let state = match entry[layer].as_mut() {
                Some(state) => state,
                None => {
                    entry[layer] = Some(self.zero_state()?);
                    entry[layer].as_mut().unwrap()
                }
            };
            for index in start..end {
                outputs.push(self.step(&projected.get(index)?, state)?);
            }
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)
    }
}

struct NemotronMlp {
    up: ReplicatedLinear,
    down: ReplicatedLinear,
}

impl NemotronMlp {
    fn new(vb: VarBuilderX, config: &Config, intermediate: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            up: ReplicatedLinear::load_no_bias(
                config.hidden_size,
                intermediate,
                vb.pp("up_proj"),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?,
            down: ReplicatedLinear::load_no_bias(
                intermediate,
                config.hidden_size,
                vb.pp("down_proj"),
                &config.quantization_config,
                &config.quant,
                dtype,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(xs)?.relu()?.sqr()?)
    }
}

struct NemotronMoe {
    gate: ReplicatedLinear,
    routing: MoeRouting,
    shared: NemotronMlp,
    experts: Vec<NemotronMlp>,
    hidden: usize,
}

impl NemotronMoe {
    fn new(vb: VarBuilderX, config: &Config, c: &NemotronConfig, dtype: DType) -> Result<Self> {
        let count = c.n_routed_experts.unwrap();
        let gate = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            count,
            vb.pp("gate"),
            &None,
            &None,
            DType::F32,
        )?;
        let bias = vb.pp("gate").get_with_hints_dtype(
            (count,),
            "e_score_correction_bias",
            candle_nn::var_builder::Shard::default(),
            DType::F32,
        )?;
        let routing = MoeRouting {
            e_score_correction_bias: Some(bias),
            use_sigmoid_scoring: true,
            n_group: c.n_group.unwrap(),
            topk_group: c.topk_group.unwrap(),
            norm_topk_prob: c.norm_topk_prob,
            routed_scaling_factor: c.routed_scaling_factor,
            num_experts_per_tok: c.num_experts_per_tok.unwrap(),
        };
        let shared = NemotronMlp::new(
            vb.pp("shared_experts"),
            config,
            c.moe_shared_expert_intermediate_size.unwrap(),
            dtype,
        )?;
        let mut experts = Vec::with_capacity(count);
        for index in 0..count {
            experts.push(NemotronMlp::new(
                vb.pp(&format!("experts.{index}")),
                config,
                c.moe_intermediate_size.unwrap(),
                dtype,
            )?);
        }
        Ok(Self {
            gate,
            routing,
            shared,
            experts,
            hidden: config.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let rows = xs.dim(0)?;
        let logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let (weights, ids) = self.routing.route(&logits, is_prefill)?;
        let ids = ids.to_vec2::<u32>()?;
        let weights = weights.to_vec2::<f32>()?;
        let mut assignments = vec![Vec::<(u32, f32)>::new(); self.experts.len()];
        for token in 0..rows {
            for (&expert, &weight) in ids[token].iter().zip(&weights[token]) {
                assignments[expert as usize].push((token as u32, weight));
            }
        }
        let mut result = Tensor::zeros((rows, self.hidden), DType::F32, xs.device())?;
        for (expert, assigned) in self.experts.iter().zip(assignments) {
            if assigned.is_empty() {
                continue;
            }
            let indices = assigned.iter().map(|(token, _)| *token).collect::<Vec<_>>();
            let factors = assigned
                .iter()
                .map(|(_, factor)| *factor)
                .collect::<Vec<_>>();
            let index_tensor = Tensor::from_vec(indices, (assigned.len(),), xs.device())?;
            let selected = xs.index_select(&index_tensor, 0)?;
            let values = expert.forward(&selected)?.to_dtype(DType::F32)?;
            let scale = Tensor::from_vec(factors, (assigned.len(), 1), xs.device())?;
            result = result.index_add(&index_tensor, &values.broadcast_mul(&scale)?, 0)?;
        }
        result
            .broadcast_add(&self.shared.forward(xs)?.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())
    }
}

enum Mixer {
    Mamba(Mamba2),
    Attention(Attention),
    Mlp(NemotronMlp),
    Moe(NemotronMoe),
}

struct Block {
    norm: NormX,
    mixer: Mixer,
}

pub struct NemotronHForCausalLM {
    embeddings: candle_nn::Embedding,
    layers: Vec<Block>,
    norm_f: NormX,
    lm_head: VocabParallelLinear,
    states: RwLock<HashMap<usize, Vec<Option<MambaState>>>>,
    prefixes: RwLock<HashMap<u64, Vec<Option<MambaState>>>>,
    dtype: DType,
    device: Device,
    vocab_size: usize,
    state_capacity: AtomicUsize,
    prefix_capacity: AtomicUsize,
}

impl NemotronHForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        _is_rope_i: bool,
        device: &Device,
        progress: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        if vb.is_qvar_builder() || comm.world_size() != 1 {
            candle_core::bail!("Nemotron-H currently requires safetensors and one device")
        }
        let c = NemotronConfig::from_config(config)?;
        // ModelOpt publishes its quantization recipe in hf_quant_config.json,
        // outside config.json. The packed weight and scale tensors identify the
        // native NVFP4 projections; the shared linear loader also detects BF16
        // projections in the same checkpoint by their missing scale tensors.
        let mut config = config.clone();
        let native_nvfp4 = (0..config.num_hidden_layers).any(|index| {
            let mixer = vb.pp(&format!("backbone.layers.{index}.mixer"));
            ["in_proj", "shared_experts.up_proj", "experts.0.up_proj"]
                .iter()
                .any(|name| mixer.pp(name).has_key("weight_scale_2"))
        });
        if config.quantization_config.is_none() && native_nvfp4 {
            let mut quant: crate::utils::config::QuantConfig = serde_json::from_value(
                serde_json::json!({"quant_method": "nvfp4", "bits": 4, "group_size": 16}),
            )
            .map_err(candle_core::Error::wrap)?;
            quant.normalize_compressed_tensors();
            config.quantization_config = Some(quant);
        }
        let (embeddings, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("backbone.embeddings"),
            dtype,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, kind) in c.hybrid_override_pattern.bytes().enumerate() {
            let lvb = vb.pp(&format!("backbone.layers.{i}"));
            let norm = rms_norm(
                config.hidden_size,
                c.layer_norm_epsilon,
                lvb.pp("norm"),
                DType::F32,
                false,
            )?;
            let mvb = lvb.pp("mixer");
            let mixer = match kind {
                b'M' => Mixer::Mamba(Mamba2::new(mvb, &config, &c, dtype)?),
                b'*' => Mixer::Attention(Attention::new(
                    mvb,
                    comm.clone(),
                    &config,
                    None,
                    None,
                    dtype,
                )?),
                b'-' => Mixer::Mlp(NemotronMlp::new(
                    mvb,
                    &config,
                    config.intermediate_size,
                    dtype,
                )?),
                b'E' => Mixer::Moe(NemotronMoe::new(mvb, &config, &c, dtype)?),
                _ => unreachable!(),
            };
            layers.push(Block { norm, mixer });
            progress.write().set_progress(i + 1);
        }
        let norm_f = rms_norm(
            config.hidden_size,
            c.layer_norm_epsilon,
            vb.pp("backbone.norm_f"),
            DType::F32,
            false,
        )?;
        let lm_head = VocabParallelLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            vb.pp("lm_head"),
            comm,
            &config.quantization_config,
            &None,
            dtype,
        )?;
        Ok(Self {
            embeddings,
            layers,
            norm_f,
            lm_head,
            states: RwLock::new(HashMap::new()),
            prefixes: RwLock::new(HashMap::new()),
            dtype,
            device: device.clone(),
            vocab_size,
            state_capacity: AtomicUsize::new(usize::MAX),
            prefix_capacity: AtomicUsize::new(0),
        })
    }

    fn spans(&self, metadata: &InputMetadata, tokens: usize) -> Result<Vec<(usize, usize, usize)>> {
        let ids = metadata
            .sequence_ids
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Nemotron-H requires sequence_ids".into()))?;
        if metadata.is_prefill {
            let ends = metadata.seqlens.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("Nemotron-H requires prefill seqlens".into())
            })?;
            if ends.len() != ids.len() || ends.last().copied().unwrap_or(0) as usize != tokens {
                candle_core::bail!("Nemotron-H prefill sequence lengths do not match tokens")
            }
            let mut start = 0;
            let mut spans = Vec::with_capacity(ids.len());
            for (&id, &end) in ids.iter().zip(ends) {
                let end = end as usize;
                if end <= start {
                    candle_core::bail!("empty Nemotron-H prefill sequence")
                }
                spans.push((id, start, end));
                start = end;
            }
            Ok(spans)
        } else {
            if ids.len() != tokens {
                candle_core::bail!("Nemotron-H decode batch mismatch")
            }
            Ok(ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i, i + 1))
                .collect())
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        metadata: &InputMetadata,
        embedded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            metadata,
            embedded_inputs,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        metadata: &InputMetadata,
        embedded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            metadata,
            embedded_inputs,
            true,
        )
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        metadata: &InputMetadata,
        embedded_inputs: bool,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let mut xs = if embedded_inputs {
            input_ids.clone()
        } else {
            self.embeddings.forward(input_ids)?
        };
        let spans = self.spans(metadata, xs.dim(0)?)?;
        let states_read = self.states.read();
        let new_count = spans
            .iter()
            .filter(|(id, _, _)| !states_read.contains_key(id))
            .count();
        if states_read.len().saturating_add(new_count) > self.state_capacity.load(Ordering::Relaxed)
        {
            candle_core::bail!("Nemotron-H recurrent state capacity exceeded")
        }
        drop(states_read);
        let mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            metadata.seqlens.clone().unwrap_or_default(),
            None,
            metadata.is_prefill,
        );
        let mut states = self.states.write();
        let mut kv_index = 0;
        for (i, block) in self.layers.iter().enumerate() {
            let input = block.norm.forward(&xs)?;
            let output = match &block.mixer {
                Mixer::Mamba(m) => m.forward(&input, &spans, &mut states, i, self.layers.len())?,
                Mixer::Attention(attn) => {
                    let cache = kv_caches.map(|all| {
                        let pair = &all[kv_index];
                        (&pair.0, &pair.1)
                    });
                    kv_index += 1;
                    attn.forward(&input, &None, mask.as_ref(), positions, cache, metadata)?
                }
                Mixer::Mlp(mlp) => mlp.forward(&input)?,
                Mixer::Moe(moe) => moe.forward(&input, metadata.is_prefill)?,
            };
            xs = (&xs + output)?;
        }
        drop(states);
        if metadata.is_prefill && !return_hidden {
            let indices = spans
                .iter()
                .map(|(_, _, end)| (*end - 1) as u32)
                .collect::<Vec<_>>();
            xs = xs.index_select(&Tensor::from_vec(indices, (spans.len(),), &self.device)?, 0)?;
        }
        let xs = self.norm_f.forward(&xs)?;
        if return_hidden {
            return xs.to_dtype(DType::F32);
        }
        self.lm_head
            .forward(&xs.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn preallocate_mamba_cache(&self, capacity: usize) -> Result<()> {
        self.state_capacity
            .store(capacity.max(1), Ordering::Relaxed);
        Ok(())
    }
    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.prefix_capacity.store(capacity, Ordering::Relaxed);
    }
    pub fn release_sequence_state(&self, seq_id: usize) {
        self.states.write().remove(&seq_id);
    }
    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.states.write().clear();
        self.prefixes.write().clear();
        Ok(())
    }
    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        _preserve: bool,
    ) -> Result<bool> {
        if self.prefix_capacity.load(Ordering::Relaxed) == 0 {
            return Ok(false);
        }
        let state = self.states.read().get(&seq_id).cloned();
        if let Some(state) = state {
            let mut prefixes = self.prefixes.write();
            if prefixes.len() >= self.prefix_capacity.load(Ordering::Relaxed)
                && !prefixes.contains_key(&hash)
            {
                if let Some(oldest) = prefixes.keys().next().copied() {
                    prefixes.remove(&oldest);
                }
            }
            prefixes.insert(hash, state);
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        let state = self.prefixes.read().get(&hash).cloned();
        if let Some(state) = state {
            self.states.write().insert(seq_id, state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Export exact FP32 Mamba state after `prefix_tokens` have been processed.
    /// The caller must export attention KV at the same token boundary and pass
    /// a SHA-256 identity for compatible weights and execution settings.
    pub fn export_mamba_state(
        &self,
        seq_id: usize,
        prefix_tokens: u64,
        model_fingerprint: [u8; 32],
    ) -> Result<NemotronMambaSnapshot> {
        let layout = state::layout_from_layers(&self.layers)?;
        // Mamba steps replace tensors instead of mutating their storage. Clone
        // the handles at the boundary, then release the map lock before the
        // device-to-host copies and checksum work.
        let sequence = self.states.read().get(&seq_id).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!("Nemotron-H sequence {seq_id} has no Mamba state"))
        })?;
        state::capture(&sequence, layout, prefix_tokens, model_fingerprint)
    }

    /// Import into an unused sequence ID. Restore attention KV from the same
    /// boundary before decoding. An existing ID is never overwritten.
    pub fn import_mamba_state(
        &self,
        seq_id: usize,
        expected_prefix_tokens: u64,
        expected_model_fingerprint: [u8; 32],
        snapshot: &NemotronMambaSnapshot,
    ) -> Result<()> {
        if self.states.read().contains_key(&seq_id) {
            candle_core::bail!("Nemotron-H sequence {seq_id} already has Mamba state")
        }
        let layout = state::layout_from_layers(&self.layers)?;
        let restored = snapshot.restore(
            &layout,
            expected_prefix_tokens,
            expected_model_fingerprint,
            &self.device,
        )?;
        state::install(
            &mut self.states.write(),
            seq_id,
            restored,
            self.state_capacity.load(Ordering::Relaxed),
        )
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.prefixes.read().contains_key(&hash)
    }
    pub fn remove_mamba_prefix_state(&self, hash: u64) -> bool {
        self.prefixes.write().remove(&hash).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_config_has_expected_checkpoint_layout() -> Result<()> {
        let raw = include_str!("../../tests/fixtures/nemotron_h_config.json");
        let mut config: Config = serde_json::from_str(raw).map_err(candle_core::Error::wrap)?;
        config.extra_config_json = Some(raw.to_string());
        let parsed = NemotronConfig::from_config(&config)?;
        assert!(matches!(
            crate::utils::config::ModelType::from_architectures(
                config.architectures.as_ref().unwrap()
            ),
            Some(crate::utils::config::ModelType::NemotronH)
        ));
        assert_eq!(parsed.hybrid_override_pattern.len(), 56);
        let types = crate::utils::qwen3_hybrid_layer_types(&config).unwrap();
        assert_eq!(
            types.iter().filter(|t| *t == "linear_attention").count(),
            27
        );
        assert_eq!(types.iter().filter(|t| *t == "mlp").count(), 25);
        let attention = types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t == "full_attention").then_some(i))
            .collect::<Vec<_>>();
        assert_eq!(attention, [14, 21, 30, 39]);
        Ok(())
    }

    #[test]
    fn nemotron_3_nano_config_has_moe_and_native_nvfp4_pattern() -> Result<()> {
        let raw = include_str!("../../tests/fixtures/nemotron_3_nano_config.json");
        let mut config: Config = serde_json::from_str(raw).map_err(candle_core::Error::wrap)?;
        config.extra_config_json = Some(raw.to_string());
        let parsed = NemotronConfig::from_config(&config)?;
        assert_eq!(parsed.hybrid_override_pattern.len(), 52);
        assert_eq!(parsed.n_routed_experts, Some(128));
        assert_eq!(parsed.num_experts_per_tok, Some(6));
        let types = crate::utils::qwen3_hybrid_layer_types(&config).unwrap();
        assert_eq!(
            types.iter().filter(|t| *t == "linear_attention").count(),
            23
        );
        assert_eq!(types.iter().filter(|t| *t == "full_attention").count(), 6);
        assert_eq!(types.iter().filter(|t| *t == "mlp").count(), 23);
        Ok(())
    }

    #[test]
    fn nemotron_moe_uses_biased_choice_and_unbiased_normalized_weights() -> Result<()> {
        let device = Device::Cpu;
        let linear = |weight: Vec<f32>, shape: (usize, usize)| {
            ReplicatedLinear::from_weight_bias(Tensor::from_vec(weight, shape, &device)?, None)
        };
        let expert = |factor: f32| -> Result<NemotronMlp> {
            Ok(NemotronMlp {
                up: linear(vec![factor, 0.0], (1, 2))?,
                down: linear(vec![1.0, 0.0], (2, 1))?,
            })
        };
        let moe = NemotronMoe {
            gate: linear(vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0], (3, 2))?,
            routing: MoeRouting {
                e_score_correction_bias: Some(Tensor::new(&[1.0f32, 0.0, 0.0], &device)?),
                use_sigmoid_scoring: true,
                n_group: 1,
                topk_group: 1,
                norm_topk_prob: true,
                routed_scaling_factor: Some(2.5),
                num_experts_per_tok: 2,
            },
            shared: expert(1.0)?,
            experts: vec![expert(1.0)?, expert(2.0)?, expert(3.0)?],
            hidden: 2,
        };
        let xs = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device)?;
        let output = moe.forward(&xs, true)?.to_vec2::<f32>()?;
        let score0 = 0.5f32;
        let score2 = 1.0 / (1.0 + (-2.0f32).exp());
        let denominator = score0 + score2;
        let expected = 1.0 + 2.5 * (score0 + 9.0 * score2) / denominator;
        assert!((output[0][0] - expected).abs() < 1e-5);
        assert_eq!(output[0][1], 0.0);
        Ok(())
    }

    #[test]
    fn mamba2_recurrence_and_causal_convolution_match_scalar_reference() -> Result<()> {
        let device = Device::Cpu;
        let tensor = |values: Vec<f32>, shape: Vec<usize>| Tensor::from_vec(values, shape, &device);
        let in_proj = ReplicatedLinear::from_weight_bias(
            tensor(vec![2.0, 3.0, 0.5, 0.25, 0.0], vec![5, 1])?,
            None,
        )?;
        let out_proj = ReplicatedLinear::from_weight_bias(tensor(vec![1.0], vec![1, 1])?, None)?;
        let mixer = Mamba2 {
            in_proj,
            out_proj,
            conv_weight: tensor(vec![0.5, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3])?,
            conv_bias: tensor(vec![0.0; 3], vec![3])?,
            dt_bias: tensor(vec![0.0], vec![1])?,
            a: tensor(vec![-1.0], vec![1])?,
            d: tensor(vec![0.1], vec![1])?,
            norm_weight: tensor(vec![1.0], vec![1])?,
            heads: 1,
            head_dim: 1,
            groups: 1,
            state_dim: 1,
            kernel: 2,
            conv_dim: 3,
            eps: 1e-5,
            dt_limit: (0.0, f64::INFINITY),
            device: device.clone(),
            dtype: DType::F32,
        };
        let projected = tensor(vec![2.0, 3.0, 0.5, 0.25, 0.0], vec![5])?;
        let mut state = mixer.zero_state()?;
        let first = mixer.step(&projected, &mut state)?;
        let x1 = 3.0f32 / (1.0 + (-3.0f32).exp());
        let b = 0.5f32 / (1.0 + (-0.5f32).exp());
        let c = 0.25f32 / (1.0 + (-0.25f32).exp());
        let dt = 2.0f32.ln();
        let s1 = x1 * dt * b;
        let actual_s1 = state.ssm.to_vec3::<f32>()?[0][0][0];
        assert!(
            (actual_s1 - s1).abs() < 1e-5,
            "actual={actual_s1} expected={s1}"
        );
        let gate = 2.0f32 / (1.0 + (-2.0f32).exp());
        let gated = (s1 * c + x1 * 0.1) * gate;
        let expected = gated / (gated * gated + 1e-5).sqrt();
        assert!((first.to_vec2::<f32>()?[0][0] - expected).abs() < 1e-5);

        mixer.step(&projected, &mut state)?;
        let x2 = 4.5f32 / (1.0 + (-4.5f32).exp());
        let s2 = 0.5 * s1 + x2 * dt * b;
        assert!((state.ssm.to_vec3::<f32>()?[0][0][0] - s2).abs() < 1e-5);
        assert_eq!(state.conv.to_vec2::<f32>()?, vec![vec![3.0, 0.5, 0.25]]);

        let mut states = HashMap::new();
        let packed = tensor(vec![1.0, 1.0, 1.0], vec![3, 1])?;
        let outputs = mixer.forward(&packed, &[(10, 0, 2), (20, 2, 3)], &mut states, 0, 1)?;
        assert_eq!(outputs.dims(), &[3, 1]);
        let for_id = |id| {
            states[&id][0]
                .as_ref()
                .unwrap()
                .ssm
                .to_vec3::<f32>()
                .unwrap()[0][0][0]
        };
        assert!((for_id(10) - s2).abs() < 1e-5);
        assert!((for_id(20) - s1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn mamba2_repeats_each_bc_group_over_its_heads() -> Result<()> {
        let device = Device::Cpu;
        let inner = 8;
        let conv_dim = 20;
        let in_proj = ReplicatedLinear::from_weight_bias(
            Tensor::zeros((32, inner), DType::F32, &device)?,
            None,
        )?;
        let mut identity = vec![0f32; inner * inner];
        for i in 0..inner {
            identity[i * inner + i] = 1.0;
        }
        let out_proj = ReplicatedLinear::from_weight_bias(
            Tensor::from_vec(identity, (inner, inner), &device)?,
            None,
        )?;
        let conv_weight = Tensor::from_vec(
            [vec![0f32; conv_dim], vec![1f32; conv_dim]].concat(),
            (2, conv_dim),
            &device,
        )?;
        let mixer = Mamba2 {
            in_proj,
            out_proj,
            conv_weight,
            conv_bias: Tensor::zeros((conv_dim,), DType::F32, &device)?,
            dt_bias: Tensor::zeros((4,), DType::F32, &device)?,
            a: Tensor::new(&[-1f32; 4], &device)?,
            d: Tensor::zeros((4,), DType::F32, &device)?,
            norm_weight: Tensor::ones((inner,), DType::F32, &device)?,
            heads: 4,
            head_dim: 2,
            groups: 2,
            state_dim: 3,
            kernel: 2,
            conv_dim,
            eps: 1e-5,
            dt_limit: (0.0, f64::INFINITY),
            device: device.clone(),
            dtype: DType::F32,
        };
        let mut projected = vec![1f32; 32];
        projected[inner + inner..inner + inner + 3].fill(0.5);
        projected[inner + inner + 3..inner + inner + 6].fill(1.0);
        projected[inner + conv_dim..].fill(0.0);
        let mut state = mixer.zero_state()?;
        let output = mixer.step(&Tensor::from_vec(projected, (32,), &device)?, &mut state)?;
        assert_eq!(output.dims(), &[1, inner]);
        let values = state.ssm.to_vec3::<f32>()?;
        assert!((values[0][0][0] - values[1][0][0]).abs() < 1e-6);
        assert!((values[2][0][0] - values[3][0][0]).abs() < 1e-6);
        assert!(values[2][0][0] > values[0][0][0]);
        Ok(())
    }
}
