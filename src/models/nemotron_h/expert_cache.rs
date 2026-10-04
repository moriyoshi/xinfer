//! Exact, byte-bounded resident cache for Nemotron-H routed experts.
use super::NemotronMlp;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{DType, Device, Result};
use candle_nn::var_builder::ShardedSafeTensors;
use either::Either;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NemotronExpertCacheStats {
    pub capacity_bytes: usize,
    /// Resident expert tensor bytes, excluding the eager gate/shared expert.
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    /// Largest logical expert allocation during a miss, before old entries
    /// are released. This excludes allocator metadata and CUDA workspaces.
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
    model: u64,
    format: String,
    device: String,
}

impl ExpertSource {
    pub(super) fn new(
        vb: &VarBuilderX,
        config: &Config,
        intermediate: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if config.quant.is_some() {
            candle_core::bail!("lazy Nemotron experts do not support runtime ISQ")
        }
        let paths: Vec<PathBuf> = vb.weight_paths().ok_or_else(|| {
            candle_core::Error::Msg("lazy Nemotron experts require safetensors paths".into())
        })?;
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
    max_seen_bytes: usize,
    stats: NemotronExpertCacheStats,
}

impl ExpertCache {
    pub(super) fn new(source: ExpertSource, capacity_bytes: usize) -> Self {
        Self {
            source,
            entries: HashMap::new(),
            clock: 0,
            max_seen_bytes: 0,
            stats: NemotronExpertCacheStats {
                capacity_bytes,
                ..Default::default()
            },
        }
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
        // Experts in a layer normally have the same shape. Reclaim space for
        // the largest expert seen so far *before* allocating the next one.
        // The first miss has an empty cache; heterogeneous larger experts are
        // still checked against the exact byte count after loading.
        if self.max_seen_bytes > 0 {
            self.make_room(self.max_seen_bytes.min(self.stats.capacity_bytes))?;
        }
        let expert = Arc::new(self.source.load(layer, id)?);
        let bytes = expert.resident_bytes()?;
        self.stats.peak_live_expert_bytes = self
            .stats
            .peak_live_expert_bytes
            .max(self.stats.resident_bytes.saturating_add(bytes));
        if bytes > self.stats.capacity_bytes {
            candle_core::bail!(
                "Nemotron expert {layer}/{id} needs {bytes} bytes, cache limit is {}",
                self.stats.capacity_bytes
            );
        }
        self.make_room(bytes)?;
        self.max_seen_bytes = self.max_seen_bytes.max(bytes);
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
        let source = ExpertSource::new(&vb, &config, 16, DType::F32, device)?;
        let cache = Arc::new(Mutex::new(ExpertCache::new(source, 2 * 1024)));
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
        let (eager, lazy, cache, path) = fixture(&device)?;
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
        cache.lock().stats.capacity_bytes = 512;
        assert!(lazy.forward(&input(&[0], &device)?, false).is_err());
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
        let (eager, lazy, cache, path) = fixture(&device)?;
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
        let mut config: Config = serde_json::from_value(raw).map_err(candle_core::Error::wrap)?;
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
        let source = ExpertSource::new(&vb, &config, 128, DType::BF16, &device)?;
        let mut cache = ExpertCache::new(source, 64 * 1024);
        let selected = cache.resolve(0, 0)?;
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
