//! Opt-in whole-model eager/lazy parity against a Nemotron-H MoE checkpoint.
//! Set XINFER_NEMOTRON_MOE_CHECKPOINT to a local safetensors model directory.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Context, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::{
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Instant,
};
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        nemotron_h::{NemotronExpertRestoreSource, NemotronHForCausalLM},
    },
    utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
};

const BLOCK: usize = 16;

fn load(
    directory: &Path,
    device: &Device,
    budget: Option<usize>,
) -> Result<(NemotronHForCausalLM, Config)> {
    let raw = std::fs::read_to_string(directory.join("config.json"))?;
    let mut config: Config = serde_json::from_str(&raw)?;
    config.extra_config_json = Some(raw);
    let mut filenames = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    filenames.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "safetensors")
    });
    filenames.sort();
    ensure!(
        !filenames.is_empty(),
        "MoE checkpoint has no safetensors shards"
    );
    let paths = ModelPaths {
        tokenizer_filename: directory.join("tokenizer.json"),
        tokenizer_config_filename: directory.join("tokenizer_config.json"),
        config_filename: directory.join("config.json"),
        generation_config_filename: directory.join("generation_config.json"),
        filenames,
        auxiliary_filenames: vec![],
        chat_template_filename: None,
    };
    let vb = VarBuilderX::new(&paths, false, DType::BF16, device)?;
    let model = NemotronHForCausalLM::new_with_expert_cache(
        &vb,
        Rc::new(Comm::default()),
        &config,
        DType::BF16,
        false,
        device,
        Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        budget,
    )?;
    Ok((model, config))
}

struct SnapshotDirectory(PathBuf);

impl SnapshotDirectory {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "xinfer-nemotron-expert-restore-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }

    fn path(&self, layer: usize, expert: usize) -> PathBuf {
        self.0.join(format!("{layer}-{expert}.xnex"))
    }
}

impl Drop for SnapshotDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct FileRestoreSource(PathBuf);

impl NemotronExpertRestoreSource for FileRestoreSource {
    fn load_expert(&self, layer: usize, expert: usize) -> candle_core::Result<Option<Vec<u8>>> {
        let path = self.0.join(format!("{layer}-{expert}.xnex"));
        match std::fs::read(path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(candle_core::Error::wrap(error)),
        }
    }
}

fn checkpoint_fingerprint(directory: &Path) -> Result<[u8; 32]> {
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    digest.update(b"xinfer-nemotron-checkpoint-v1");
    digest.update(&std::fs::read(directory.join("config.json"))?);
    let mut shards = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    shards.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    shards.sort();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for path in shards {
        digest.update(path.file_name().unwrap().as_encoded_bytes());
        let mut file = std::fs::File::open(path)?;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(digest.finish().as_ref().try_into()?)
}

fn cache(config: &Config, device: &Device) -> Result<Vec<(Tensor, Tensor)>> {
    let raw: serde_json::Value = serde_json::from_str(config.extra_config_json.as_ref().unwrap())?;
    let attention_layers = raw["hybrid_override_pattern"]
        .as_str()
        .context("layer pattern")?
        .bytes()
        .filter(|&layer| layer == b'*')
        .count();
    let shape = (
        4,
        BLOCK,
        config.num_key_value_heads,
        config.head_dim.context("head dim")?,
    );
    (0..attention_layers)
        .map(|_| {
            Ok((
                Tensor::zeros(shape, DType::BF16, device)?,
                Tensor::zeros(shape, DType::BF16, device)?,
            ))
        })
        .collect()
}

fn forward(
    model: &NemotronHForCausalLM,
    tokens: &[u32],
    start: usize,
    kv: &Vec<(Tensor, Tensor)>,
    device: &Device,
) -> Result<Vec<f32>> {
    let prefill = start == 0;
    let _guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
    let end = start + tokens.len();
    let positions = (start..end).map(|i| i as i64).collect::<Vec<_>>();
    let block_ids = (0..end.div_ceil(BLOCK) as u32).collect::<Vec<_>>();
    let metadata = InputMetadata {
        is_prefill: prefill,
        is_mla: false,
        sequence_ids: Some(vec![0]),
        mamba_slot_mapping: None,
        slot_mapping: Tensor::from_vec(positions.clone(), tokens.len(), device)?,
        block_tables: Some(Tensor::from_vec(
            block_ids.clone(),
            (1, block_ids.len()),
            device,
        )?),
        block_tables_host: Some(vec![block_ids]),
        context_lens_host: Some(vec![end as u32]),
        context_lens: Some(Tensor::from_vec(vec![end as u32], 1, device)?),
        cu_seqlens_q: if prefill {
            Some(Tensor::from_vec(
                vec![0u32, tokens.len() as u32],
                2,
                device,
            )?)
        } else {
            None
        },
        cu_seqlens_k: if prefill {
            Some(Tensor::from_vec(vec![0u32, end as u32], 2, device)?)
        } else {
            None
        },
        max_seqlen_q: if prefill { tokens.len() } else { 0 },
        max_seqlen_k: if prefill { end } else { 0 },
        max_context_len: end,
        seqlens: if prefill {
            Some(vec![tokens.len() as u32])
        } else {
            None
        },
        flashinfer_metadata: None,
        is_mtp_verify: false,
    };
    Ok(model
        .forward(
            &Tensor::from_vec(tokens.to_vec(), tokens.len(), device)?,
            &Tensor::from_vec(positions, tokens.len(), device)?,
            Some(kv),
            &metadata,
            false,
        )?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

#[test]
fn whole_model_eager_and_selective_experts_match() -> Result<()> {
    let Some(directory) = std::env::var_os("XINFER_NEMOTRON_MOE_CHECKPOINT") else {
        return Ok(());
    };
    let directory = std::path::PathBuf::from(directory);
    let budget = std::env::var("XINFER_NEMOTRON_MOE_CACHE_BYTES")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(256 * 1024);
    let device = Device::new_cuda(0)?;
    if let Ok(mode) = std::env::var("XINFER_NEMOTRON_MOE_MEMORY_ONLY") {
        ensure!(
            mode == "eager" || mode == "lazy",
            "invalid memory-only mode"
        );
        let before = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?.0;
        let (model, _) = load(&directory, &device, (mode == "lazy").then_some(budget))?;
        device.synchronize()?;
        let after = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?.0;
        eprintln!(
            "Nemotron MoE {mode} cold model CUDA used_delta={} B, logical_expert_bytes={} B",
            before.saturating_sub(after),
            model.resident_routed_expert_bytes()?
        );
        return Ok(());
    }
    let free_before = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _)| free);
    let (eager, config) = load(&directory, &device, None)?;
    device.synchronize()?;
    let free_after_eager = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _)| free);
    let (lazy, _) = load(&directory, &device, Some(budget))?;
    device.synchronize()?;
    let free_after_lazy = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _)| free);
    let eager_kv = cache(&config, &device)?;
    let lazy_kv = cache(&config, &device)?;
    let prefix = [1, 123, 456, 789, 42, 43, 44, 45];
    let mut max_abs = 0.0f32;
    let mut compare = |a: Vec<f32>, b: Vec<f32>| -> Result<()> {
        ensure!(a.len() == b.len(), "logit count changed");
        for (left, right) in a.into_iter().zip(b) {
            max_abs = max_abs.max((left - right).abs());
        }
        Ok(())
    };
    compare(
        forward(&eager, &prefix, 0, &eager_kv, &device)?,
        forward(&lazy, &prefix, 0, &lazy_kv, &device)?,
    )?;
    device.synchronize()?;
    let free_after_prefill = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _)| free);
    let mut eager_samples = Vec::new();
    let mut samples = Vec::new();
    for i in 0..16 {
        let token = [42 + (i % 4) as u32];
        let eager_start = Instant::now();
        let expected = forward(&eager, &token, prefix.len() + i, &eager_kv, &device)?;
        eager_samples.push(eager_start.elapsed().as_secs_f64() * 1000.0);
        let start = Instant::now();
        let actual = forward(&lazy, &token, prefix.len() + i, &lazy_kv, &device)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        compare(expected, actual)?;
    }
    eager_samples.sort_by(f64::total_cmp);
    samples.sort_by(f64::total_cmp);
    let stats = lazy
        .expert_cache_stats()
        .context("selective cache missing")?;
    let eager_expert_bytes = eager.resident_routed_expert_bytes()?;
    if let (Some(before), Some(after_eager), Some(after_lazy), Some(after_prefill)) = (
        free_before,
        free_after_eager,
        free_after_lazy,
        free_after_prefill,
    ) {
        eprintln!("CUDA free bytes: before={before}, after_eager={after_eager}, after_lazy={after_lazy}, after_prefill={after_prefill}; eager_load_delta={}, lazy_load_delta={}, lazy_prefill_delta={}",
            before.saturating_sub(after_eager), after_eager.saturating_sub(after_lazy),
            after_lazy.saturating_sub(after_prefill));
    }
    eprintln!("Nemotron MoE eager/lazy max_abs={max_abs}, eager_expert_bytes={eager_expert_bytes} B, lazy_resident={} B, lazy_peak={} B, lazy_live_peak={} B, transferred_estimate={} B, hits={}, misses={}, evictions={}, eager_p50={:.3} ms, eager_p95={:.3} ms, lazy_p50={:.3} ms, lazy_p95={:.3} ms",
        stats.resident_bytes, stats.peak_resident_bytes, stats.peak_live_expert_bytes, stats.transferred_bytes,
        stats.hits, stats.misses, stats.evictions, eager_samples[8], eager_samples[15], samples[8], samples[15]);
    ensure!(max_abs <= 1e-4, "selective expert loading changed logits");
    ensure!(
        stats.misses > 0
            && stats.resident_bytes <= budget
            && stats.peak_live_expert_bytes <= budget,
        "cache was not exercised within budget"
    );
    Ok(())
}

/// Run explicitly with `-- --ignored --nocapture`; writes temporary per-expert
/// snapshots, then compares three sequential model instances on one GB10.
#[test]
#[ignore = "requires the real 30B checkpoint and several GiB of temporary snapshot storage"]
fn real_checkpoint_persistent_expert_restore_benchmark() -> Result<()> {
    let directory = std::env::var_os("XINFER_NEMOTRON_MOE_CHECKPOINT")
        .context("set XINFER_NEMOTRON_MOE_CHECKPOINT to the real 30B checkpoint")?;
    let directory = PathBuf::from(directory);
    let budget = 128 * 1024 * 1024;
    let device = Device::new_cuda(0)?;
    let free = || -> Result<usize> {
        device.synchronize()?;
        Ok(candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?.0)
    };
    let run =
        |model: &NemotronHForCausalLM, config: &Config| -> Result<(Vec<Vec<f32>>, Vec<f64>)> {
            let kv = cache(config, &device)?;
            let prefix = [1, 123, 456, 789, 42, 43, 44, 45];
            let mut outputs = vec![forward(model, &prefix, 0, &kv, &device)?];
            let mut times = Vec::with_capacity(16);
            for i in 0..16 {
                let token = [42 + (i % 4) as u32];
                let started = Instant::now();
                outputs.push(forward(model, &token, prefix.len() + i, &kv, &device)?);
                times.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            Ok((outputs, times))
        };
    let compare = |expected: &[Vec<f32>], actual: &[Vec<f32>]| -> Result<f32> {
        ensure!(
            expected.len() == actual.len(),
            "different number of continuation steps"
        );
        let mut max_abs = 0.0f32;
        for (left, right) in expected.iter().zip(actual) {
            ensure!(left.len() == right.len(), "different logit vector width");
            for (&a, &b) in left.iter().zip(right) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        Ok(max_abs)
    };

    let free_before_eager = free()?;
    let (eager, config) = load(&directory, &device, None)?;
    let free_after_eager = free()?;
    let eager_expert_bytes = eager.resident_routed_expert_bytes()?;
    let (expected, eager_times) = run(&eager, &config)?;
    drop(eager);

    let free_before_checkpoint = free()?;
    let (checkpoint, checkpoint_config) = load(&directory, &device, Some(budget))?;
    let free_after_checkpoint = free()?;
    let (checkpoint_outputs, checkpoint_times) = run(&checkpoint, &checkpoint_config)?;
    let checkpoint_max_abs = compare(&expected, &checkpoint_outputs)?;
    let checkpoint_stats = checkpoint
        .expert_cache_stats()
        .context("missing checkpoint cache")?;
    ensure!(checkpoint_max_abs == 0.0, "checkpoint-lazy logits changed");

    // Hash the actual checkpoint bytes outside the timed inference windows.
    let fingerprint = checkpoint_fingerprint(&directory)?;
    let scratch = SnapshotDirectory::new()?;
    let raw: serde_json::Value =
        serde_json::from_str(checkpoint_config.extra_config_json.as_ref().unwrap())?;
    let count = raw["n_routed_experts"].as_u64().context("expert count")? as usize;
    let pattern = raw["hybrid_override_pattern"]
        .as_str()
        .context("layer pattern")?;
    let export_started = Instant::now();
    let mut snapshot_count = 0usize;
    let mut snapshot_bytes = 0usize;
    for (layer, kind) in pattern.bytes().enumerate() {
        if kind != b'E' {
            continue;
        }
        for expert in 0..count {
            let bytes = checkpoint.export_expert_snapshot_bytes(layer, expert, fingerprint)?;
            snapshot_bytes += bytes.len();
            std::fs::write(scratch.path(layer, expert), bytes)?;
            snapshot_count += 1;
        }
    }
    let export_seconds = export_started.elapsed().as_secs_f64();
    drop(checkpoint);

    let free_before_restored = free()?;
    let (restored, restored_config) = load(&directory, &device, Some(budget))?;
    restored
        .set_expert_restore_source(fingerprint, Arc::new(FileRestoreSource(scratch.0.clone())))?;
    let free_after_restored = free()?;
    let (restored_outputs, restored_times) = run(&restored, &restored_config)?;
    let restored_max_abs = compare(&expected, &restored_outputs)?;
    let restored_stats = restored
        .expert_cache_stats()
        .context("missing restored cache")?;
    ensure!(restored_max_abs == 0.0, "restored-lazy logits changed");
    ensure!(
        restored_stats.restored_loads > 0,
        "restore source was not exercised"
    );
    ensure!(
        restored_stats.checkpoint_loads == 0,
        "restore fell back to checkpoint"
    );
    ensure!(
        checkpoint_stats.peak_live_expert_bytes <= budget
            && restored_stats.peak_live_expert_bytes <= budget,
        "expert GPU budget exceeded"
    );
    let avg_ms = |total_ns: u64, loads: u64| total_ns as f64 / loads.max(1) as f64 / 1e6;
    eprintln!(
        "real Nemotron expert restore: snapshots={snapshot_count}, disk_bytes={snapshot_bytes}, export_s={export_seconds:.3}, checkpoint_max_abs={checkpoint_max_abs}, restored_max_abs={restored_max_abs}, eager_expert_bytes={eager_expert_bytes}, checkpoint_peak={}, restored_peak={}, checkpoint_loads={}, restored_loads={}, checkpoint_load_avg_ms={:.3}, restored_load_avg_ms={:.3}, eager_p50_ms={:.3}, eager_p95_ms={:.3}, checkpoint_p50_ms={:.3}, checkpoint_p95_ms={:.3}, restored_p50_ms={:.3}, restored_p95_ms={:.3}, eager_cuda_load_delta={}, checkpoint_cuda_load_delta={}, restored_cuda_load_delta={}",
        checkpoint_stats.peak_live_expert_bytes,
        restored_stats.peak_live_expert_bytes,
        checkpoint_stats.checkpoint_loads,
        restored_stats.restored_loads,
        avg_ms(checkpoint_stats.checkpoint_load_ns, checkpoint_stats.checkpoint_loads),
        avg_ms(restored_stats.restored_load_ns, restored_stats.restored_loads),
        eager_times[8], eager_times[15],
        checkpoint_times[8], checkpoint_times[15],
        restored_times[8], restored_times[15],
        free_before_eager.saturating_sub(free_after_eager),
        free_before_checkpoint.saturating_sub(free_after_checkpoint),
        free_before_restored.saturating_sub(free_after_restored),
    );
    drop(restored);

    let free_before_host = free()?;
    let (host, host_config) = load(&directory, &device, Some(budget))?;
    let import_started = Instant::now();
    for (layer, kind) in pattern.bytes().enumerate() {
        if kind != b'E' {
            continue;
        }
        for expert in 0..count {
            let bytes = std::fs::read(scratch.path(layer, expert))?;
            host.import_expert_snapshot_bytes(fingerprint, &bytes)?;
        }
    }
    let import_seconds = import_started.elapsed().as_secs_f64();
    let free_after_host = free()?;
    let (host_outputs, host_times) = run(&host, &host_config)?;
    let host_max_abs = compare(&expected, &host_outputs)?;
    let host_stats = host.expert_cache_stats().context("missing host cache")?;
    ensure!(host_max_abs == 0.0, "host-restored logits changed");
    ensure!(host_stats.checkpoint_loads == 0 && host_stats.restored_loads > 0);
    ensure!(host_stats.peak_live_expert_bytes <= budget);
    eprintln!(
        "real Nemotron host restore: host_snapshot_bytes={}, import_s={import_seconds:.3}, max_abs={host_max_abs}, restored_loads={}, restored_load_avg_ms={:.3}, token_p50_ms={:.3}, token_p95_ms={:.3}, cuda_load_delta={}",
        host_stats.host_snapshot_bytes,
        host_stats.restored_loads,
        avg_ms(host_stats.restored_load_ns, host_stats.restored_loads),
        host_times[8], host_times[15],
        free_before_host.saturating_sub(free_after_host),
    );
    Ok(())
}
