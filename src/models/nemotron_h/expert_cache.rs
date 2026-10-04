//! Exact, byte-bounded resident cache for Nemotron-H routed experts.
use super::expert_snapshot::{
    NemotronExpertLayout, NemotronExpertSnapshot, NemotronExpertTensorDType,
    NemotronExpertTensorSpec, NEMOTRON_EXPERT_SNAPSHOT_VERSION,
};
use super::{NemotronConfig, NemotronMlp};
use crate::models::layers::distributed::ReplicatedLinear;
use crate::models::layers::linear::{LinearX, LnNvfp4};
use crate::models::layers::state_bytes;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Result, Tensor};
use candle_nn::var_builder::ShardedSafeTensors;
use either::Either;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Optional application-owned persistent source for expert snapshots. Return
/// `None` to fall back to the original safetensors checkpoint. A returned
/// snapshot is always validated before any GPU upload; invalid data is an
/// error, not a silent checkpoint fallback.
pub trait NemotronExpertRestoreSource: Send + Sync {
    fn load_expert(&self, layer: usize, expert: usize) -> Result<Option<Vec<u8>>>;
}

#[derive(Clone)]
struct TensorMeta {
    shape: Vec<usize>,
    dtype: String,
    bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct WeightPlan {
    /// Final device tensors held by the expert.
    resident_bytes: usize,
    /// Maximum expert weight bytes during its sequential projection loads.
    admission_bytes: usize,
}

fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron expert byte count overflow".into()))
}

fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| candle_core::Error::Msg("Nemotron expert byte count overflow".into()))
}

fn tensor<'a>(
    metadata: &'a HashMap<String, TensorMeta>,
    name: &str,
    shape: &[usize],
) -> Result<&'a TensorMeta> {
    let meta = metadata
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing Nemotron expert tensor {name}")))?;
    if meta.shape != shape {
        candle_core::bail!(
            "Nemotron expert tensor {name} has shape {:?}, expected {shape:?}",
            meta.shape
        )
    }
    Ok(meta)
}

fn dense_projection(
    metadata: &HashMap<String, TensorMeta>,
    prefix: &str,
    input: usize,
    output: usize,
    dtype: DType,
) -> Result<WeightPlan> {
    let meta = tensor(metadata, &format!("{prefix}.weight"), &[output, input])?;
    let resident_bytes = checked_mul(checked_mul(input, output)?, dtype.size_in_bytes())?;
    let admission_bytes = if meta.dtype == format!("{dtype:?}") {
        resident_bytes
    } else {
        // The dense loader first uploads the checkpoint tensor, then casts it.
        checked_add(meta.bytes, resident_bytes)?
    };
    Ok(WeightPlan {
        resident_bytes,
        admission_bytes,
    })
}

fn nvfp4_projection(
    metadata: &HashMap<String, TensorMeta>,
    prefix: &str,
    input: usize,
    output: usize,
    sm_version: usize,
) -> Result<WeightPlan> {
    if input % 16 != 0 {
        candle_core::bail!("Nemotron NVFP4 expert {prefix} input width must be divisible by 16")
    }
    let packed_name = ["weight_packed", "weight", "blocks"]
        .into_iter()
        .find(|name| metadata.contains_key(&format!("{prefix}.{name}")))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("missing packed NVFP4 weight for {prefix}"))
        })?;
    let packed = tensor(
        metadata,
        &format!("{prefix}.{packed_name}"),
        &[output, input / 2],
    )?;
    if packed.dtype != "U8" {
        candle_core::bail!("Nemotron NVFP4 expert {prefix} weight must be packed U8")
    }
    let scale_name = ["weight_scale", "scales"]
        .into_iter()
        .find(|name| metadata.contains_key(&format!("{prefix}.{name}")))
        .ok_or_else(|| candle_core::Error::Msg(format!("missing NVFP4 scales for {prefix}")))?;
    let scales = tensor(
        metadata,
        &format!("{prefix}.{scale_name}"),
        &[output, input / 16],
    )?;
    if !matches!(scales.dtype.as_str(), "U8" | "F8_E4M3") {
        candle_core::bail!("Nemotron NVFP4 expert {prefix} scales must be U8 or F8_E4M3")
    }
    for name in [
        "weight_scale_2",
        "weight_global_scale",
        "input_scale",
        "input_global_scale",
    ] {
        if let Some(scalar) = metadata.get(&format!("{prefix}.{name}")) {
            if scalar.dtype != "F32" || (!scalar.shape.is_empty() && scalar.shape != [1]) {
                candle_core::bail!("Nemotron NVFP4 expert {prefix}.{name} must be an F32 scalar")
            }
        }
    }
    let swizzled_bytes = if sm_version >= 100 {
        let rows = checked_mul(output.div_ceil(128), 128)?;
        let cols = checked_mul((input / 16).div_ceil(4), 4)?;
        checked_mul(rows, cols)?
    } else {
        0
    };
    let raw_bytes = checked_add(packed.bytes, scales.bytes)?;
    let resident_bytes = checked_add(raw_bytes, swizzled_bytes)?;
    // A U8 scale may be temporarily converted when the native loader first
    // probes F8E4M3; allow its exact encoded size in admission headroom.
    let scale_cast_peak = checked_add(
        raw_bytes,
        if scales.dtype == "U8" {
            scales.bytes
        } else {
            0
        },
    )?;
    let scalar_peak = checked_add(raw_bytes, DType::F32.size_in_bytes())?;
    let admission_bytes = resident_bytes.max(scale_cast_peak).max(scalar_peak);
    Ok(WeightPlan {
        resident_bytes,
        admission_bytes,
    })
}

fn expert_plan(
    metadata: &HashMap<String, TensorMeta>,
    config: &Config,
    layer: usize,
    expert: usize,
    intermediate: usize,
    dtype: DType,
    sm_version: usize,
) -> Result<WeightPlan> {
    let prefix = format!("backbone.layers.{layer}.mixer.experts.{expert}");
    let projection = |name: &str, input: usize, output: usize| -> Result<WeightPlan> {
        let module = format!("{prefix}.{name}");
        let native_nvfp4 = config.quantization_config.as_ref().is_some_and(|quant| {
            quant.quant_method == "nvfp4"
                && !quant.should_skip_module(&module)
                && [
                    "weight_packed",
                    "blocks",
                    "weight_scale_2",
                    "weight_global_scale",
                ]
                .iter()
                .any(|name| metadata.contains_key(&format!("{module}.{name}")))
        });
        if native_nvfp4 {
            nvfp4_projection(metadata, &module, input, output, sm_version)
        } else {
            if config
                .quantization_config
                .as_ref()
                .is_some_and(|quant| quant.quant_method == "nvfp4")
                && ["weight_scale", "weight_scale_inv", "scale", "scales"]
                    .iter()
                    .any(|name| metadata.contains_key(&format!("{module}.{name}")))
                && !config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .should_skip_module(&module)
            {
                candle_core::bail!("unsupported quantized Nemotron expert projection {module}")
            }
            dense_projection(metadata, &module, input, output, dtype)
        }
    };
    let up = projection("up_proj", config.hidden_size, intermediate)?;
    let down = projection("down_proj", intermediate, config.hidden_size)?;
    Ok(WeightPlan {
        resident_bytes: checked_add(up.resident_bytes, down.resident_bytes)?,
        admission_bytes: up
            .admission_bytes
            .max(checked_add(up.resident_bytes, down.admission_bytes)?),
    })
}

fn model_layout_digest(
    metadata: &HashMap<String, TensorMeta>,
    config: &Config,
    nemotron: &NemotronConfig,
    dtype: DType,
) -> [u8; 32] {
    fn field(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    }
    let mut canonical = Vec::with_capacity(metadata.len() * 96);
    field(&mut canonical, b"xinfer-nemotron-expert-layout-v1");
    field(&mut canonical, nemotron.hybrid_override_pattern.as_bytes());
    field(&mut canonical, format!("{dtype:?}").as_bytes());
    field(&mut canonical, &config.hidden_size.to_le_bytes());
    field(&mut canonical, &config.num_hidden_layers.to_le_bytes());
    field(
        &mut canonical,
        config
            .quantization_config
            .as_ref()
            .map(|quant| quant.quant_method.as_bytes())
            .unwrap_or(b"dense"),
    );
    let mut names = metadata.keys().collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        let tensor = &metadata[name];
        field(&mut canonical, name.as_bytes());
        field(&mut canonical, tensor.dtype.as_bytes());
        field(&mut canonical, &(tensor.shape.len() as u64).to_le_bytes());
        for &dim in &tensor.shape {
            field(&mut canonical, &(dim as u64).to_le_bytes());
        }
        field(&mut canonical, &(tensor.bytes as u64).to_le_bytes());
    }
    state_bytes::sha256(&canonical)
}

fn expert_layout(
    metadata: &HashMap<String, TensorMeta>,
    model_layout_sha256: [u8; 32],
    layer: usize,
    expert: usize,
    dtype: DType,
    format: &str,
) -> Result<NemotronExpertLayout> {
    let prefix = format!("backbone.layers.{layer}.mixer.experts.{expert}.");
    let mut tensors = metadata
        .iter()
        .filter_map(|(name, meta)| name.strip_prefix(&prefix).map(|suffix| (suffix, meta)))
        .map(|(name, meta)| {
            Ok(NemotronExpertTensorSpec {
                name: name.to_owned(),
                dtype: NemotronExpertTensorDType::from_safetensors(&meta.dtype)?,
                shape: meta
                    .shape
                    .iter()
                    .map(|&dim| {
                        u32::try_from(dim).map_err(|_| {
                            candle_core::Error::Msg("Nemotron expert dimension exceeds u32".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                byte_len: meta.bytes as u64,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(NemotronExpertLayout {
        model_layout_sha256,
        layer: u32::try_from(layer)
            .map_err(|_| candle_core::Error::Msg("Nemotron layer exceeds u32".into()))?,
        expert: u32::try_from(expert)
            .map_err(|_| candle_core::Error::Msg("Nemotron expert exceeds u32".into()))?,
        activation_dtype: format!("{dtype:?}"),
        quant_format: format.to_owned(),
        tensors,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NemotronExpertCacheStats {
    pub capacity_bytes: usize,
    /// Resident expert tensor bytes, excluding the eager gate/shared expert.
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    /// Host payload bytes retained by explicit snapshot imports. Excludes
    /// application-owned restore sources and GPU cache entries.
    pub host_snapshot_bytes: usize,
    /// Largest preflight estimate of live expert weight tensors during a miss.
    /// This excludes allocator metadata and CUDA workspaces.
    pub peak_live_expert_bytes: usize,
    /// Estimated host-to-device payload; derived/swizzled tensors make this
    /// an upper estimate for some native quantized projections.
    pub transferred_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// GPU misses served by safetensors and by validated host snapshots.
    pub checkpoint_loads: u64,
    pub restored_loads: u64,
    /// Total wall time spent loading one expert, including the restore-source
    /// callback where applicable. Divide by the corresponding load count.
    pub checkpoint_load_ns: u64,
    pub restored_load_ns: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ExpertKey {
    model: u64,
    layer: usize,
    expert: usize,
    format: String,
    device: String,
}

/// The safetensors backend owns its mmaps; loading a projection through this
/// builder copies precisely that projection's original tensors to the device.
/// Keeping the backend here makes the host source independent of the caller's
/// VarBuilder lifetime.
pub(super) struct ExpertSource {
    vb: VarBuilderX<'static>,
    checkpoint: MmapedSafetensors,
    config: Config,
    intermediate: usize,
    dtype: DType,
    plans: HashMap<(usize, usize), WeightPlan>,
    layouts: HashMap<(usize, usize), NemotronExpertLayout>,
    model: u64,
    format: String,
    device: String,
    device_handle: Device,
    sm_version: usize,
}

impl ExpertSource {
    pub(super) fn new(
        vb: &VarBuilderX,
        config: &Config,
        nemotron: &NemotronConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if config.quant.is_some() {
            candle_core::bail!("lazy Nemotron experts do not support runtime ISQ")
        }
        if let Some(quant) = &config.quantization_config {
            if quant.quant_method != "nvfp4" || quant.is_mlx_nvfp4 {
                candle_core::bail!(
                    "lazy Nemotron experts support dense and native NVFP4 safetensors only"
                )
            }
        }
        let paths: Vec<PathBuf> = vb.weight_paths().ok_or_else(|| {
            candle_core::Error::Msg("lazy Nemotron experts require safetensors paths".into())
        })?;
        // Candle's mmap reader parses and validates the safetensors headers;
        // views below borrow file bytes without uploading tensor payloads.
        let checkpoint = unsafe { MmapedSafetensors::multi(&paths)? };
        let metadata = checkpoint
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    name,
                    TensorMeta {
                        shape: view.shape().to_vec(),
                        dtype: format!("{:?}", view.dtype()),
                        bytes: view.data().len(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let sm_version = {
            #[cfg(feature = "cuda")]
            {
                if let Device::Cuda(cuda) = device {
                    attention_rs::cuda_utils::sm_version(cuda).unwrap_or(0) as usize
                } else {
                    0
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                0
            }
        };
        let intermediate = nemotron.moe_intermediate_size.unwrap();
        let count = nemotron.n_routed_experts.unwrap();
        let format = config
            .quantization_config
            .as_ref()
            .map(|q| q.quant_method.clone())
            .unwrap_or_else(|| "dense".into());
        let model_layout_sha256 = model_layout_digest(&metadata, config, nemotron, dtype);
        let mut plans = HashMap::new();
        let mut layouts = HashMap::new();
        for (layer, kind) in nemotron.hybrid_override_pattern.bytes().enumerate() {
            if kind == b'E' {
                for expert in 0..count {
                    plans.insert(
                        (layer, expert),
                        expert_plan(
                            &metadata,
                            config,
                            layer,
                            expert,
                            intermediate,
                            dtype,
                            sm_version,
                        )?,
                    );
                    layouts.insert(
                        (layer, expert),
                        expert_layout(
                            &metadata,
                            model_layout_sha256,
                            layer,
                            expert,
                            dtype,
                            &format,
                        )?,
                    );
                }
            }
        }
        drop(metadata);
        let backend = unsafe { ShardedSafeTensors::var_builder(&paths, dtype, device)? };
        Ok(Self {
            vb: VarBuilderX(
                Either::Left(backend),
                String::new(),
                None,
                None,
                Some(paths),
            ),
            checkpoint,
            config: config.clone(),
            intermediate,
            dtype,
            plans,
            layouts,
            model: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            format: format!("{format}:{dtype:?}"),
            device: format!("{device:?}"),
            device_handle: device.clone(),
            sm_version,
        })
    }

    fn key(&self, layer: usize, expert: usize) -> ExpertKey {
        ExpertKey {
            model: self.model,
            layer,
            expert,
            format: self.format.clone(),
            device: self.device.clone(),
        }
    }

    fn load(&self, layer: usize, expert: usize) -> Result<NemotronMlp> {
        NemotronMlp::new(
            self.vb
                .pp(&format!("backbone.layers.{layer}.mixer.experts.{expert}")),
            &self.config,
            self.intermediate,
            self.dtype,
        )
    }

    fn plan(&self, layer: usize, expert: usize) -> Result<WeightPlan> {
        self.plans.get(&(layer, expert)).copied().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "missing Nemotron expert metadata for {layer}/{expert}"
            ))
        })
    }

    fn layout(&self, layer: usize, expert: usize) -> Result<&NemotronExpertLayout> {
        self.layouts.get(&(layer, expert)).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "missing Nemotron expert layout for {layer}/{expert}"
            ))
        })
    }

    fn export(
        &self,
        layer: usize,
        expert: usize,
        model_fingerprint: [u8; 32],
    ) -> Result<NemotronExpertSnapshot> {
        if model_fingerprint == [0; 32] {
            candle_core::bail!("Nemotron expert snapshot requires a nonzero model fingerprint")
        }
        let layout = self.layout(layer, expert)?.clone();
        let prefix = format!("backbone.layers.{layer}.mixer.experts.{expert}.");
        let capacity = layout.tensors.iter().try_fold(0usize, |total, spec| {
            total.checked_add(spec.byte_len as usize).ok_or_else(|| {
                candle_core::Error::Msg("Nemotron expert payload byte overflow".into())
            })
        })?;
        let mut payload = Vec::with_capacity(capacity);
        for spec in &layout.tensors {
            let view = self.checkpoint.get(&format!("{prefix}{}", spec.name))?;
            if view.data().len() != spec.byte_len as usize {
                candle_core::bail!("Nemotron expert checkpoint tensor changed during export")
            }
            payload.extend_from_slice(view.data());
        }
        let snapshot = NemotronExpertSnapshot {
            version: NEMOTRON_EXPERT_SNAPSHOT_VERSION,
            model_fingerprint,
            layout,
            payload_sha256: state_bytes::sha256(&payload),
            payload,
        };
        Ok(snapshot)
    }

    fn validate_snapshot(
        &self,
        snapshot: &NemotronExpertSnapshot,
        model_fingerprint: [u8; 32],
    ) -> Result<(usize, usize)> {
        snapshot.validate()?;
        self.validate_snapshot_identity(snapshot, model_fingerprint)
    }

    /// `from_bytes` already checked the payload checksum.
    fn validate_snapshot_identity(
        &self,
        snapshot: &NemotronExpertSnapshot,
        model_fingerprint: [u8; 32],
    ) -> Result<(usize, usize)> {
        if model_fingerprint == [0; 32] || snapshot.model_fingerprint != model_fingerprint {
            candle_core::bail!("Nemotron expert model fingerprint mismatch")
        }
        let layer = snapshot.layout.layer as usize;
        let expert = snapshot.layout.expert as usize;
        if self.layout(layer, expert)? != &snapshot.layout {
            candle_core::bail!("Nemotron expert model, shape, or format layout mismatch")
        }
        Ok((layer, expert))
    }

    fn load_restored(&self, snapshot: &NemotronExpertSnapshot) -> Result<NemotronMlp> {
        let layer = snapshot.layout.layer as usize;
        let expert = snapshot.layout.expert as usize;
        let prefix = format!("backbone.layers.{layer}.mixer.experts.{expert}");
        let up = self.restore_projection(
            snapshot,
            &prefix,
            "up_proj",
            self.config.hidden_size,
            self.intermediate,
        )?;
        let down = self.restore_projection(
            snapshot,
            &prefix,
            "down_proj",
            self.intermediate,
            self.config.hidden_size,
        )?;
        Ok(NemotronMlp { up, down })
    }

    fn restore_projection(
        &self,
        snapshot: &NemotronExpertSnapshot,
        prefix: &str,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<ReplicatedLinear> {
        let module = format!("{prefix}.{name}");
        let tensor = |suffix: &str| snapshot.tensor(&format!("{name}.{suffix}"));
        let upload = |suffix: &str| -> Result<Tensor> {
            let (spec, bytes) = tensor(suffix)?;
            let shape = spec
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .collect::<Vec<_>>();
            Tensor::from_raw_buffer(bytes, spec.dtype.candle(), &shape, &self.device_handle)
        };
        let native_nvfp4 = self
            .config
            .quantization_config
            .as_ref()
            .is_some_and(|quant| {
                quant.quant_method == "nvfp4"
                    && !quant.should_skip_module(&module)
                    && [
                        "weight_packed",
                        "blocks",
                        "weight_scale_2",
                        "weight_global_scale",
                    ]
                    .iter()
                    .any(|suffix| snapshot.has_tensor(&format!("{name}.{suffix}")))
            });
        if !native_nvfp4 {
            let weight = upload("weight")?.to_dtype(self.dtype)?;
            if weight.dims() != [output, input] {
                candle_core::bail!("restored dense Nemotron expert shape mismatch")
            }
            return ReplicatedLinear::from_weight_bias(weight, None);
        }
        let packed_name = ["weight_packed", "weight", "blocks"]
            .into_iter()
            .find(|suffix| snapshot.has_tensor(&format!("{name}.{suffix}")))
            .ok_or_else(|| candle_core::Error::Msg("missing restored NVFP4 blocks".into()))?;
        let scale_name = ["weight_scale", "scales"]
            .into_iter()
            .find(|suffix| snapshot.has_tensor(&format!("{name}.{suffix}")))
            .ok_or_else(|| candle_core::Error::Msg("missing restored NVFP4 scales".into()))?;
        let blocks = upload(packed_name)?;
        let scales = upload(scale_name)?;
        if blocks.dtype() != DType::U8
            || blocks.dims() != [output, input / 2]
            || scales.dims() != [output, input / 16]
            || !matches!(scales.dtype(), DType::F8E4M3 | DType::U8)
        {
            candle_core::bail!("restored NVFP4 Nemotron expert shape or dtype mismatch")
        }
        let scalar = |suffix: &str| -> Result<f32> {
            let (_, bytes) = tensor(suffix)?;
            if bytes.len() != 4 {
                candle_core::bail!("restored NVFP4 scalar {suffix} has invalid length")
            }
            Ok(f32::from_bits(u32::from_le_bytes(
                bytes.try_into().unwrap(),
            )))
        };
        let global_scale = if snapshot.has_tensor(&format!("{name}.weight_global_scale")) {
            let value = scalar("weight_global_scale")?;
            if value == 0.0 {
                1.0
            } else {
                1.0 / value
            }
        } else if snapshot.has_tensor(&format!("{name}.weight_scale_2")) {
            scalar("weight_scale_2")?
        } else {
            1.0
        };
        let input_scale = if snapshot.has_tensor(&format!("{name}.input_scale")) {
            scalar("input_scale")?
        } else if snapshot.has_tensor(&format!("{name}.input_global_scale")) {
            let value = scalar("input_global_scale")?;
            if value == 0.0 {
                1.0
            } else {
                1.0 / value
            }
        } else {
            1.0
        };
        #[cfg(feature = "cuda")]
        let weight_scale_swizzled = if self.sm_version >= 100 {
            Some(attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                &scales,
            )?)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let weight_scale_swizzled = None;
        ReplicatedLinear::from(LinearX::LnNvfp4(LnNvfp4 {
            blocks,
            scales,
            global_scale,
            input_scale,
            bias: None,
            weight_scale_swizzled,
        }))
    }
}

struct CacheEntry {
    expert: Arc<NemotronMlp>,
    bytes: usize,
    last_use: u64,
}

pub(super) struct ExpertCache {
    source: ExpertSource,
    entries: HashMap<ExpertKey, CacheEntry>,
    restored: HashMap<(usize, usize), Arc<NemotronExpertSnapshot>>,
    restore_source: Option<Arc<dyn NemotronExpertRestoreSource>>,
    restore_model_fingerprint: Option<[u8; 32]>,
    clock: u64,
    stats: NemotronExpertCacheStats,
}

impl ExpertCache {
    pub(super) fn new(source: ExpertSource, capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            candle_core::bail!("Nemotron expert cache capacity must be positive")
        }
        if let Some((&(layer, expert), plan)) = source
            .plans
            .iter()
            .find(|(_, plan)| plan.admission_bytes > capacity_bytes)
        {
            candle_core::bail!("Nemotron expert {layer}/{expert} needs {} bytes during loading, cache limit is {capacity_bytes}", plan.admission_bytes)
        }
        Ok(Self {
            source,
            entries: HashMap::new(),
            restored: HashMap::new(),
            restore_source: None,
            restore_model_fingerprint: None,
            clock: 0,
            stats: NemotronExpertCacheStats {
                capacity_bytes,
                ..Default::default()
            },
        })
    }

    pub(super) fn stats(&self) -> NemotronExpertCacheStats {
        self.stats.clone()
    }

    pub(super) fn export(
        &self,
        layer: usize,
        expert: usize,
        model_fingerprint: [u8; 32],
    ) -> Result<NemotronExpertSnapshot> {
        self.source.export(layer, expert, model_fingerprint)
    }

    fn bind_fingerprint(&mut self, fingerprint: [u8; 32]) -> Result<()> {
        if fingerprint == [0; 32] {
            candle_core::bail!("Nemotron expert restore requires a nonzero model fingerprint")
        }
        if self
            .restore_model_fingerprint
            .is_some_and(|bound| bound != fingerprint)
        {
            candle_core::bail!("Nemotron expert restore model fingerprint changed")
        }
        self.restore_model_fingerprint = Some(fingerprint);
        Ok(())
    }

    pub(super) fn import(
        &mut self,
        expected_fingerprint: [u8; 32],
        snapshot: NemotronExpertSnapshot,
    ) -> Result<()> {
        let key = self
            .source
            .validate_snapshot(&snapshot, expected_fingerprint)?;
        self.install_validated(key, expected_fingerprint, snapshot)
    }

    pub(super) fn import_bytes(
        &mut self,
        expected_fingerprint: [u8; 32],
        bytes: &[u8],
    ) -> Result<()> {
        let snapshot = NemotronExpertSnapshot::from_bytes(bytes)?;
        let key = self
            .source
            .validate_snapshot_identity(&snapshot, expected_fingerprint)?;
        self.install_validated(key, expected_fingerprint, snapshot)
    }

    fn install_validated(
        &mut self,
        key: (usize, usize),
        expected_fingerprint: [u8; 32],
        snapshot: NemotronExpertSnapshot,
    ) -> Result<()> {
        if self.restored.contains_key(&key) {
            candle_core::bail!(
                "Nemotron expert snapshot already imported for {}/{}",
                key.0,
                key.1
            )
        }
        let host_snapshot_bytes = self
            .stats
            .host_snapshot_bytes
            .checked_add(snapshot.payload.len())
            .ok_or_else(|| {
                candle_core::Error::Msg("Nemotron host snapshot byte overflow".into())
            })?;
        self.bind_fingerprint(expected_fingerprint)?;
        self.stats.host_snapshot_bytes = host_snapshot_bytes;
        self.restored.insert(key, Arc::new(snapshot));
        Ok(())
    }

    pub(super) fn set_restore_source(
        &mut self,
        expected_fingerprint: [u8; 32],
        source: Arc<dyn NemotronExpertRestoreSource>,
    ) -> Result<()> {
        if self.restore_source.is_some() {
            candle_core::bail!("Nemotron expert restore source already installed")
        }
        self.bind_fingerprint(expected_fingerprint)?;
        self.restore_source = Some(source);
        Ok(())
    }

    fn make_room(&mut self, bytes: usize) -> Result<()> {
        while self.stats.resident_bytes > self.stats.capacity_bytes - bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, entry)| Arc::strong_count(&entry.expert) == 1)
                .min_by_key(|(_, entry)| entry.last_use)
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else {
                candle_core::bail!("Nemotron expert cache is full of active experts");
            };
            let old = self.entries.remove(&victim).unwrap();
            self.stats.resident_bytes -= old.bytes;
            self.stats.evictions += 1;
        }
        Ok(())
    }

    /// The returned Arc pins this expert while its assigned tokens execute.
    pub(super) fn resolve(&mut self, layer: usize, id: usize) -> Result<Arc<NemotronMlp>> {
        let key = self.source.key(layer, id);
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_use = self.clock;
            self.stats.hits += 1;
            return Ok(entry.expert.clone());
        }

        self.stats.misses += 1;
        let plan = self.source.plan(layer, id)?;
        let started = Instant::now();
        let restored = if let Some(snapshot) = self.restored.get(&(layer, id)) {
            Some(snapshot.clone())
        } else if let Some(store) = &self.restore_source {
            match store.load_expert(layer, id)? {
                Some(bytes) => {
                    let snapshot = NemotronExpertSnapshot::from_bytes(&bytes)?;
                    self.source.validate_snapshot_identity(
                        &snapshot,
                        self.restore_model_fingerprint.unwrap(),
                    )?;
                    if snapshot.layout.layer as usize != layer
                        || snapshot.layout.expert as usize != id
                    {
                        candle_core::bail!(
                            "Nemotron expert restore source returned the wrong expert"
                        )
                    }
                    Some(Arc::new(snapshot))
                }
                None => None,
            }
        } else {
            None
        };
        self.make_room(plan.admission_bytes)?;
        self.stats.peak_live_expert_bytes = self.stats.peak_live_expert_bytes.max(
            self.stats
                .resident_bytes
                .saturating_add(plan.admission_bytes),
        );
        let expert = Arc::new(if let Some(snapshot) = &restored {
            self.source.load_restored(snapshot)?
        } else {
            self.source.load(layer, id)?
        });
        let bytes = expert.resident_bytes()?;
        if bytes != plan.resident_bytes {
            candle_core::bail!(
                "Nemotron expert {layer}/{id} loaded {bytes} resident bytes, metadata predicted {}",
                plan.resident_bytes
            )
        }
        self.stats.resident_bytes += bytes;
        self.stats.peak_resident_bytes = self
            .stats
            .peak_resident_bytes
            .max(self.stats.resident_bytes);
        self.stats.transferred_bytes += bytes as u64;
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if restored.is_some() {
            self.stats.restored_loads += 1;
            self.stats.restored_load_ns = self.stats.restored_load_ns.saturating_add(elapsed_ns);
        } else {
            self.stats.checkpoint_loads += 1;
            self.stats.checkpoint_load_ns =
                self.stats.checkpoint_load_ns.saturating_add(elapsed_ns);
        }
        self.entries.insert(
            key,
            CacheEntry {
                expert: expert.clone(),
                bytes,
                last_use: self.clock,
            },
        );
        Ok(expert)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{NemotronConfig, NemotronMoe};
    use super::*;
    use candle_core::Tensor;
    use parking_lot::Mutex;
    use serde_json::Value;

    fn fixture(
        device: &Device,
        capacity_bytes: usize,
    ) -> Result<(
        NemotronMoe,
        NemotronMoe,
        Arc<Mutex<ExpertCache>>,
        PathBuf,
        Config,
    )> {
        let mut raw: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/nemotron_3_nano_config.json"
        ))
        .map_err(candle_core::Error::wrap)?;
        raw["hidden_size"] = 8.into();
        raw["intermediate_size"] = 16.into();
        raw["num_hidden_layers"] = 1.into();
        raw["hybrid_override_pattern"] = "E".into();
        raw["n_routed_experts"] = 4.into();
        raw["num_experts_per_tok"] = 1.into();
        raw["moe_intermediate_size"] = 16.into();
        raw["moe_shared_expert_intermediate_size"] = 16.into();
        let mut config: Config =
            serde_json::from_value(raw.clone()).map_err(candle_core::Error::wrap)?;
        config.extra_config_json = Some(raw.to_string());
        let c = NemotronConfig::from_config(&config)?;
        let mut tensors = HashMap::new();
        let prefix = "backbone.layers.0.mixer";
        let gate = (0..4)
            .flat_map(|row| (0..8).map(move |col| if row == col { 10.0f32 } else { 0.0 }))
            .collect::<Vec<_>>();
        tensors.insert(
            format!("{prefix}.gate.weight"),
            Tensor::from_vec(gate, (4, 8), &Device::Cpu)?,
        );
        tensors.insert(
            format!("{prefix}.gate.e_score_correction_bias"),
            Tensor::zeros(4, DType::F32, &Device::Cpu)?,
        );
        for expert in 0..5 {
            let name = if expert == 4 {
                "shared_experts".to_string()
            } else {
                format!("experts.{expert}")
            };
            let up = vec![0.25f32; 16 * 8];
            let down = vec![
                if expert == 4 {
                    0.0
                } else {
                    (expert + 1) as f32 * 0.125
                };
                8 * 16
            ];
            tensors.insert(
                format!("{prefix}.{name}.up_proj.weight"),
                Tensor::from_vec(up, (16, 8), &Device::Cpu)?,
            );
            tensors.insert(
                format!("{prefix}.{name}.down_proj.weight"),
                Tensor::from_vec(down, (8, 16), &Device::Cpu)?,
            );
        }
        let path =
            std::env::temp_dir().join(format!("xinfer-moe-{}.safetensors", uuid::Uuid::new_v4()));
        candle_core::safetensors::save(&tensors, &path)?;
        let backend = unsafe { ShardedSafeTensors::var_builder(&[&path], DType::F32, device)? };
        let vb = VarBuilderX(
            Either::Left(backend),
            String::new(),
            None,
            None,
            Some(vec![path.clone()]),
        );
        let source = ExpertSource::new(&vb, &config, &c, DType::F32, device)?;
        let cache = match ExpertCache::new(source, capacity_bytes) {
            Ok(cache) => Arc::new(Mutex::new(cache)),
            Err(error) => {
                std::fs::remove_file(&path).map_err(candle_core::Error::wrap)?;
                return Err(error);
            }
        };
        let prefix_vb = vb.pp(prefix);
        let eager = NemotronMoe::new(prefix_vb.clone(), &config, &c, DType::F32, 0, None)?;
        let lazy = NemotronMoe::new(prefix_vb, &config, &c, DType::F32, 0, Some(&cache))?;
        Ok((eager, lazy, cache, path, config))
    }

    fn input(ids: &[usize], device: &Device) -> Result<Tensor> {
        let mut values = vec![0f32; ids.len() * 8];
        for (row, &id) in ids.iter().enumerate() {
            values[row * 8 + id] = 1.0;
        }
        Tensor::from_vec(values, (ids.len(), 8), device)
    }

    fn compare(
        eager: &NemotronMoe,
        lazy: &NemotronMoe,
        ids: &[usize],
        device: &Device,
    ) -> Result<()> {
        let x = input(ids, device)?;
        let expected = eager
            .forward(&x, ids.len() > 1)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = lazy
            .forward(&x, ids.len() > 1)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn exact_routing_cache_cold_warm_eviction_and_batch() -> Result<()> {
        let device = Device::Cpu;
        let (eager, lazy, cache, path, _) = fixture(&device, 2 * 1024)?;
        compare(&eager, &lazy, &[0, 1], &device)?;
        let stats = cache.lock().stats();
        assert_eq!((stats.hits, stats.misses), (0, 2));
        compare(&eager, &lazy, &[0, 1], &device)?;
        let stats = cache.lock().stats();
        assert_eq!((stats.hits, stats.misses), (2, 2));
        compare(&eager, &lazy, &[2], &device)?;
        assert_eq!(cache.lock().stats().evictions, 1);
        compare(&eager, &lazy, &[0], &device)?;
        assert_eq!(cache.lock().stats().misses, 4);
        assert!(cache.lock().stats().resident_bytes <= 2 * 1024);
        assert_eq!(cache.lock().stats().transferred_bytes, 4 * 1024);
        // Four simultaneous sequences touch four experts with room for two.
        // Streaming them in native order still produces the eager output.
        compare(&eager, &lazy, &[0, 1, 2, 3], &device)?;
        assert!(cache.lock().stats().resident_bytes <= 2 * 1024);
        assert!(cache.lock().stats().peak_live_expert_bytes <= 2 * 1024);
        drop((eager, lazy, cache));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[test]
    fn impossible_budget_is_rejected_before_expert_load() {
        let error = fixture(&Device::Cpu, 1023)
            .err()
            .expect("undersized budget must fail");
        assert!(format!("{error}").contains("needs 1024 bytes during loading"));
    }

    #[test]
    fn cross_instance_snapshot_restore_validates_identity_and_keeps_budget() -> Result<()> {
        let device = Device::Cpu;
        let (eager, _, source_cache, path, config) = fixture(&device, 1024)?;
        let fingerprint = [7; 32];
        let snapshot = source_cache.lock().export(0, 0, fingerprint)?;
        let bytes = snapshot.to_bytes()?;
        let decoded = NemotronExpertSnapshot::from_bytes(&bytes)?;
        assert_eq!(decoded.payload, snapshot.payload);

        let c = NemotronConfig::from_config(&config)?;
        let backend = unsafe { ShardedSafeTensors::var_builder(&[&path], DType::F32, &device)? };
        let vb = VarBuilderX(
            Either::Left(backend),
            String::new(),
            None,
            None,
            Some(vec![path.clone()]),
        );
        let source = ExpertSource::new(&vb, &config, &c, DType::F32, &device)?;
        let mut target_cache = ExpertCache::new(source, 1024)?;
        assert!(target_cache.import([8; 32], decoded.clone()).is_err());
        let mut wrong = decoded.clone();
        wrong.layout.model_layout_sha256[0] ^= 1;
        assert!(target_cache.import(fingerprint, wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.layout.quant_format = "nvfp4".into();
        assert!(target_cache.import(fingerprint, wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.layout.tensors[0].shape = vec![1, 1];
        assert!(target_cache.import(fingerprint, wrong).is_err());
        let mut wrong = decoded.clone();
        wrong.payload[0] ^= 1;
        assert!(target_cache.import(fingerprint, wrong).is_err());
        assert!(NemotronExpertSnapshot::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        target_cache.import(fingerprint, decoded.clone())?;
        assert!(target_cache.import(fingerprint, decoded).is_err());
        let target_cache = Arc::new(Mutex::new(target_cache));
        let restored = NemotronMoe::new(
            vb.pp("backbone.layers.0.mixer"),
            &config,
            &c,
            DType::F32,
            0,
            Some(&target_cache),
        )?;
        compare(&eager, &restored, &[0, 1, 2, 3], &device)?;
        compare(&eager, &restored, &[0], &device)?;
        let stats = target_cache.lock().stats();
        assert!(stats.restored_loads >= 2);
        assert!(stats.checkpoint_loads >= 3);
        assert!(stats.peak_live_expert_bytes <= 1024);
        drop((restored, target_cache, source_cache, eager, vb));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[test]
    fn restore_source_serves_selected_expert_and_falls_back_exactly() -> Result<()> {
        struct HostStore(Vec<u8>);
        impl NemotronExpertRestoreSource for HostStore {
            fn load_expert(&self, layer: usize, expert: usize) -> Result<Option<Vec<u8>>> {
                Ok((layer == 0 && expert == 0).then(|| self.0.clone()))
            }
        }
        let device = Device::Cpu;
        let (eager, lazy, cache, path, _) = fixture(&device, 1024)?;
        let fingerprint = [9; 32];
        let bytes = cache.lock().export(0, 0, fingerprint)?.to_bytes()?;
        cache
            .lock()
            .set_restore_source(fingerprint, Arc::new(HostStore(bytes)))?;
        compare(&eager, &lazy, &[0, 1], &device)?;
        let stats = cache.lock().stats();
        assert_eq!(stats.restored_loads, 1);
        assert_eq!(stats.checkpoint_loads, 1);
        assert_eq!(stats.misses, 2);
        assert!(stats.peak_live_expert_bytes <= 1024);
        drop((eager, lazy, cache));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn exact_routing_on_cuda_and_token_latency() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (eager, lazy, cache, path, _) = fixture(&device, 2 * 1024)?;
        compare(&eager, &lazy, &[0, 1, 2, 3], &device)?;
        let mut samples = Vec::new();
        for _ in 0..32 {
            let start = std::time::Instant::now();
            let output = lazy.forward(&input(&[0], &device)?, false)?;
            let _ = output
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        let stats = cache.lock().stats();
        eprintln!(
            "synthetic Nemotron MoE GPU: resident={} B, transferred_estimate={} B, hits={}, misses={}, evictions={}, p50={:.3} ms, p95={:.3} ms",
            stats.resident_bytes, stats.transferred_bytes, stats.hits, stats.misses,
            stats.evictions, samples[16], samples[30]
        );
        drop((eager, lazy, cache));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_nvfp4_expert_keeps_packed_weights_and_scales() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let mut raw: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/nemotron_3_nano_config.json"
        ))
        .map_err(candle_core::Error::wrap)?;
        raw["hidden_size"] = 128.into();
        raw["moe_intermediate_size"] = 128.into();
        raw["num_hidden_layers"] = 1.into();
        raw["hybrid_override_pattern"] = "E".into();
        raw["n_routed_experts"] = 1.into();
        raw["num_experts_per_tok"] = 1.into();
        let mut config: Config =
            serde_json::from_value(raw.clone()).map_err(candle_core::Error::wrap)?;
        config.extra_config_json = Some(raw.to_string());
        let c = NemotronConfig::from_config(&config)?;
        let mut quant: crate::utils::config::QuantConfig = serde_json::from_value(
            serde_json::json!({"quant_method":"nvfp4","bits":4,"group_size":16}),
        )
        .map_err(candle_core::Error::wrap)?;
        quant.normalize_compressed_tensors();
        config.quantization_config = Some(quant);
        let mut tensors = HashMap::new();
        for projection in ["up_proj", "down_proj"] {
            let prefix = format!("backbone.layers.0.mixer.experts.0.{projection}");
            tensors.insert(
                format!("{prefix}.weight_packed"),
                Tensor::from_vec(vec![0x11u8; 128 * 64], (128, 64), &Device::Cpu)?,
            );
            tensors.insert(
                format!("{prefix}.weight_scale"),
                Tensor::from_vec(vec![0x38u8; 128 * 8], (128, 8), &Device::Cpu)?,
            );
            tensors.insert(
                format!("{prefix}.weight_scale_2"),
                Tensor::from_vec(vec![1.0f32], (1,), &Device::Cpu)?,
            );
        }
        let path = std::env::temp_dir().join(format!(
            "xinfer-moe-fp4-{}.safetensors",
            uuid::Uuid::new_v4()
        ));
        candle_core::safetensors::save(&tensors, &path)?;
        let backend = unsafe { ShardedSafeTensors::var_builder(&[&path], DType::BF16, &device)? };
        let vb = VarBuilderX(
            Either::Left(backend),
            String::new(),
            None,
            None,
            Some(vec![path.clone()]),
        );
        let eager = NemotronMlp::new(
            vb.pp("backbone.layers.0.mixer.experts.0"),
            &config,
            128,
            DType::BF16,
        )?;
        let source = ExpertSource::new(&vb, &config, &c, DType::BF16, &device)?;
        let expected = source.plan(0, 0)?;
        assert!(ExpertCache::new(source, expected.admission_bytes - 1).is_err());
        let source = ExpertSource::new(&vb, &config, &c, DType::BF16, &device)?;
        let mut cache = ExpertCache::new(source, 64 * 1024)?;
        let selected = cache.resolve(0, 0)?;
        assert_eq!(selected.resident_bytes()?, expected.resident_bytes);
        assert!(cache.stats().peak_live_expert_bytes <= 64 * 1024);
        assert_eq!(selected.resident_bytes()?, eager.resident_bytes()?);
        assert!(selected.resident_bytes()? < 2 * 128 * 128 * DType::BF16.size_in_bytes());
        let input = Tensor::ones((1, 128), DType::BF16, &device)?;
        let expected = eager
            .forward(&input)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        let actual = selected
            .forward(&input)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        assert_eq!(expected, actual);
        let snapshot = cache.source.export(0, 0, [11; 32])?;
        assert!(snapshot
            .layout
            .tensors
            .iter()
            .any(|tensor| tensor.dtype == NemotronExpertTensorDType::U8));
        let encoded = snapshot.to_bytes()?;
        let source = ExpertSource::new(&vb, &config, &c, DType::BF16, &device)?;
        let mut restored_cache = ExpertCache::new(source, 64 * 1024)?;
        restored_cache.import([11; 32], NemotronExpertSnapshot::from_bytes(&encoded)?)?;
        let restored = restored_cache.resolve(0, 0)?;
        assert_eq!(restored.resident_bytes()?, selected.resident_bytes()?);
        let restored_output = restored
            .forward(&input)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        assert_eq!(expected, restored_output);
        assert_eq!(restored_cache.stats().restored_loads, 1);
        assert_eq!(restored_cache.stats().checkpoint_loads, 0);
        drop((restored, restored_cache, selected, eager, cache, vb));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
