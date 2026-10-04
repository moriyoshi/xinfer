//! Opt-in whole-model eager/lazy parity against a Nemotron-H MoE checkpoint.
//! Set XINFER_NEMOTRON_MOE_CHECKPOINT to a local safetensors model directory.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Context, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        nemotron_h::NemotronHForCausalLM,
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
    let path = directory.join("model.safetensors");
    ensure!(path.exists(), "single-shard MoE checkpoint required");
    let paths = ModelPaths {
        tokenizer_filename: directory.join("tokenizer.json"),
        tokenizer_config_filename: directory.join("tokenizer_config.json"),
        config_filename: directory.join("config.json"),
        generation_config_filename: directory.join("generation_config.json"),
        filenames: vec![path],
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
        stats.misses > 0 && stats.resident_bytes <= budget,
        "cache was not exercised within budget"
    );
    Ok(())
}
