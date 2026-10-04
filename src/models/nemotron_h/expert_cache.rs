//! Exact, byte-bounded resident cache for Nemotron-H routed experts.
use super::{NemotronConfig, NemotronMlp};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Result};
use candle_nn::var_builder::ShardedSafeTensors;
use either::Either;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NemotronExpertCacheStats {
    pub capacity_bytes: usize,
    /// Resident expert tensor bytes, excluding the eager gate/shared expert.
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    /// Largest preflight estimate of live expert weight tensors during a miss.
    /// This excludes allocator metadata and CUDA workspaces.
    pub peak_live_expert_bytes: usize,
    /// Estimated host-to-device payload; derived/swizzled tensors make this
    /// an upper estimate for some native quantized projections.
    pub transferred_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
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
    config: Config,
    intermediate: usize,
    dtype: DType,
    plans: HashMap<(usize, usize), WeightPlan>,
    model: u64,
    format: String,
    device: String,
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
        let mut plans = HashMap::new();
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
                }
            }
        }
        drop((metadata, checkpoint));
        let backend = unsafe { ShardedSafeTensors::var_builder(&paths, dtype, device)? };
        let format = config
            .quantization_config
            .as_ref()
            .map(|q| q.quant_method.clone())
            .unwrap_or_else(|| "dense".into());
        Ok(Self {
            vb: VarBuilderX(
                Either::Left(backend),
                String::new(),
                None,
                None,
                Some(paths),
            ),
            config: config.clone(),
            intermediate,
            dtype,
            plans,
            model: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            format: format!("{format}:{dtype:?}"),
            device: format!("{device:?}"),
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
}

struct CacheEntry {
    expert: Arc<NemotronMlp>,
    bytes: usize,
    last_use: u64,
}

pub(super) struct ExpertCache {
    source: ExpertSource,
    entries: HashMap<ExpertKey, CacheEntry>,
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
        self.make_room(plan.admission_bytes)?;
        self.stats.peak_live_expert_bytes = self.stats.peak_live_expert_bytes.max(
            self.stats
                .resident_bytes
                .saturating_add(plan.admission_bytes),
        );
        let expert = Arc::new(self.source.load(layer, id)?);
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
    ) -> Result<(NemotronMoe, NemotronMoe, Arc<Mutex<ExpertCache>>, PathBuf)> {
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
        Ok((eager, lazy, cache, path))
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
        let (eager, lazy, cache, path) = fixture(&device, 2 * 1024)?;
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

    #[cfg(feature = "cuda")]
    #[test]
    fn exact_routing_on_cuda_and_token_latency() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (eager, lazy, cache, path) = fixture(&device, 2 * 1024)?;
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
        drop((selected, eager, cache, vb));
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
