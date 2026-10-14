//! Opt-in real-checkpoint Mamba prefill benchmark and numerical comparison.
//! XINFER_NEMOTRON_PREFILL_CHECKPOINT points at the 30B NVFP4 checkpoint.
//! XINFER_NEMOTRON_PREFILL_TOKENS selects 2048 (default) or 32768 tokens.
//! XINFER_NEMOTRON_BENCH_CHUNK overrides the default 8192-token call boundary.
//! XINFER_NEMOTRON_BENCH_KV_BLOCK overrides the default 16-token KV page size.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Context, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path, rc::Rc, sync::Arc, time::Instant};
use tokenizers::Tokenizer;
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        nemotron_h::NemotronHForCausalLM,
    },
    utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
};

const CHUNK: usize = 8192;
type Cache = Vec<(Tensor, Tensor)>;

fn block_size() -> Result<usize> {
    let size = std::env::var("XINFER_NEMOTRON_BENCH_KV_BLOCK")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    ensure!(
        size.is_power_of_two(),
        "benchmark KV block must be a power of two"
    );
    Ok(size)
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

fn load(directory: &Path, device: &Device) -> Result<(NemotronHForCausalLM, Config)> {
    let raw = std::fs::read_to_string(directory.join("config.json"))?;
    let mut config: Config = serde_json::from_str(&raw)?;
    config.extra_config_json = Some(raw.clone());
    let mut filenames = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    filenames.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    filenames.sort();
    ensure!(!filenames.is_empty(), "checkpoint has no safetensors files");
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
        None,
    )?;
    Ok((model, config))
}

fn cache(config: &Config, capacity: usize, device: &Device) -> Result<Cache> {
    let block = block_size()?;
    let raw: serde_json::Value = serde_json::from_str(
        config
            .extra_config_json
            .as_ref()
            .context("missing original config")?,
    )?;
    let layers = raw["hybrid_override_pattern"]
        .as_str()
        .context("missing hybrid pattern")?
        .bytes()
        .filter(|&kind| kind == b'*')
        .count();
    let shape = (
        capacity.div_ceil(block),
        block,
        config.num_key_value_heads,
        config.head_dim.context("missing head_dim")?,
    );
    (0..layers)
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
    device: &Device,
    kv: &Cache,
    seq_id: usize,
    tokens: &[u32],
    start: usize,
) -> Result<Vec<f32>> {
    let block = block_size()?;
    let prefill = start == 0 || tokens.len() > 1;
    let _guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
    let end = start + tokens.len();
    let positions = (start..end).map(|i| i as i64).collect::<Vec<_>>();
    let block_ids = (0..end.div_ceil(block) as u32).collect::<Vec<_>>();
    let metadata = InputMetadata {
        is_prefill: prefill,
        is_mla: false,
        sequence_ids: Some(vec![seq_id]),
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
        cu_seqlens_q: prefill
            .then(|| Tensor::from_vec(vec![0u32, tokens.len() as u32], 2, device))
            .transpose()?,
        cu_seqlens_k: prefill
            .then(|| Tensor::from_vec(vec![0u32, end as u32], 2, device))
            .transpose()?,
        max_seqlen_q: if prefill { tokens.len() } else { 0 },
        max_seqlen_k: if prefill { end } else { 0 },
        max_context_len: end,
        seqlens: prefill.then_some(vec![tokens.len() as u32]),
        flashinfer_metadata: None,
        is_mtp_verify: false,
    };
    let logits = model.forward(
        &Tensor::from_vec(tokens.to_vec(), tokens.len(), device)?,
        &Tensor::from_vec(positions, tokens.len(), device)?,
        Some(kv),
        &metadata,
        false,
    )?;
    device.synchronize()?;
    Ok(logits.flatten_all()?.to_vec1::<f32>()?)
}

fn prefill(
    model: &NemotronHForCausalLM,
    device: &Device,
    kv: &Cache,
    seq_id: usize,
    tokens: &[u32],
    fast: bool,
) -> Result<(Vec<f32>, f64)> {
    model.set_mamba_prefill_fast_enabled(fast);
    let chunk = std::env::var("XINFER_NEMOTRON_BENCH_CHUNK")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(CHUNK);
    ensure!(chunk > 0, "benchmark prefill chunk must be positive");
    let began = Instant::now();
    let profile = std::env::var_os("XINFER_NEMOTRON_CUDA_PROFILE").is_some();
    if profile {
        ensure!(
            unsafe { cudaProfilerStart() } == 0,
            "CUDA profiler start failed"
        );
    }
    let mut last = Vec::new();
    for start in (0..tokens.len()).step_by(chunk) {
        let end = (start + chunk).min(tokens.len());
        last = forward(model, device, kv, seq_id, &tokens[start..end], start)?;
    }
    if profile {
        ensure!(
            unsafe { cudaProfilerStop() } == 0,
            "CUDA profiler stop failed"
        );
    }
    Ok((last, began.elapsed().as_secs_f64() * 1000.0))
}

fn top1(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0
}

fn logit_bytes(logits: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for &value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn real_30b_moe_prefill_and_decode_benchmark() -> Result<()> {
    if std::env::var_os("XINFER_NEMOTRON_MOE_BENCH").is_none() {
        return Ok(());
    }
    let directory = std::env::var_os("XINFER_NEMOTRON_PREFILL_CHECKPOINT")
        .context("XINFER_NEMOTRON_PREFILL_CHECKPOINT is required")?;
    let directory = Path::new(&directory);
    let length = std::env::var("XINFER_NEMOTRON_PREFILL_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2048);
    let steps = std::env::var("XINFER_NEMOTRON_MOE_BENCH_STEPS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(256);
    ensure!(steps > 0, "benchmark requires at least one decode step");
    if let Some(path) = std::env::var_os("XINFER_NEMOTRON_ROUTING_TRACE") {
        std::fs::File::create(path)?;
    }
    let device = Device::new_cuda(0)?;
    let free_before = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?.0;
    let began = Instant::now();
    let (model, config) = load(directory, &device)?;
    device.synchronize()?;
    let free_after = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?.0;
    eprintln!(
        "model_gpu_used_delta_bytes={}",
        free_before.saturating_sub(free_after)
    );
    model.set_mamba_decode_fused_enabled(
        std::env::var_os("XINFER_NEMOTRON_FUSED_DECODE").as_deref()
            != Some(std::ffi::OsStr::new("0")),
    );
    eprintln!(
        "model_load_ms={:.3}",
        began.elapsed().as_secs_f64() * 1000.0
    );
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let sample = tokenizer
        .encode(
            "A long document can mix history, engineering, mathematics, and everyday language. The model must continue each sentence from its exact preceding context. ",
            false,
        )
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut tokens = sample
        .iter()
        .copied()
        .cycle()
        .take(length + steps)
        .collect::<Vec<_>>();
    if let Some(path) = std::env::var_os("XINFER_NEMOTRON_INPUT_IDS_FILE") {
        let prefix: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            prefix.len() == length,
            "benchmark input IDs length {} differs from requested {length}",
            prefix.len()
        );
        tokens[..length].copy_from_slice(&prefix);
    }
    let warmup_len = length.min(512);
    let warmup_kv = cache(&config, warmup_len, &device)?;
    forward(&model, &device, &warmup_kv, 999, &tokens[..warmup_len], 0)?;
    model.release_sequence_state(999);
    let kv = cache(&config, length + steps, &device)?;
    let (prefix_logits, prefill_ms) = prefill(&model, &device, &kv, 0, &tokens[..length], true)?;
    eprintln!(
        "prefill_tokens={length} prefill_ms={prefill_ms:.3} prefill_ms_per_token={:.6}",
        prefill_ms / length as f64
    );
    let check_repeat = std::env::var_os("XINFER_NEMOTRON_CHECK_DETERMINISM").is_some();
    let state_file = std::env::var_os("XINFER_NEMOTRON_STATE_FILE");
    let prefix_state = (check_repeat || state_file.is_some())
        .then(|| model.export_mamba_state_bytes(0, length as u64, [0; 32]))
        .transpose()?;
    if let Some(path) = state_file {
        std::fs::write(path, prefix_state.as_ref().unwrap())?;
    }
    let mut output = std::env::var_os("XINFER_NEMOTRON_LOGITS_FILE")
        .map(std::fs::File::create)
        .transpose()?
        .map(std::io::BufWriter::new);
    if let Some(file) = output.as_mut() {
        file.write_all(&logit_bytes(&prefix_logits))?;
    }
    let mut times = Vec::with_capacity(steps);
    let mut hashes = Vec::with_capacity(steps);
    for index in 0..steps {
        let now = Instant::now();
        let logits = forward(
            &model,
            &device,
            &kv,
            0,
            &tokens[length + index..length + index + 1],
            length + index,
        )?;
        times.push(now.elapsed().as_secs_f64() * 1000.0);
        let bytes = logit_bytes(&logits);
        if let Some(file) = output.as_mut() {
            file.write_all(&bytes)?;
        }
        if check_repeat {
            hashes.push(Sha256::digest(&bytes).to_vec());
        }
    }
    if let Some(file) = output.as_mut() {
        file.flush()?;
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "decode_steps={steps} decode_p50_ms={:.3} decode_p95_ms={:.3} decode_mean_ms={:.3}",
        times[steps / 2],
        times[(steps * 95 / 100).min(steps - 1)],
        times.iter().sum::<f64>() / steps as f64
    );
    if check_repeat {
        let end_state = model.export_mamba_state_bytes(0, (length + steps) as u64, [0; 32])?;
        model.release_sequence_state(0);
        let repeat_kv = cache(&config, length + steps, &device)?;
        let (repeat_prefix, _) = prefill(&model, &device, &repeat_kv, 0, &tokens[..length], true)?;
        ensure!(
            prefix_logits == repeat_prefix,
            "repeat prefill logits differ"
        );
        ensure!(
            prefix_state.unwrap() == model.export_mamba_state_bytes(0, length as u64, [0; 32])?,
            "repeat prefill state differs"
        );
        for index in 0..steps {
            let logits = forward(
                &model,
                &device,
                &repeat_kv,
                0,
                &tokens[length + index..length + index + 1],
                length + index,
            )?;
            ensure!(
                Sha256::digest(logit_bytes(&logits)).as_slice() == hashes[index],
                "repeat decode logits differ at {index}"
            );
        }
        ensure!(
            end_state == model.export_mamba_state_bytes(0, (length + steps) as u64, [0; 32])?,
            "repeat decode state differs"
        );
        eprintln!("repeat_prefill_decode_logits_and_states_bit_identical=true");
    }
    Ok(())
}

#[test]
fn real_30b_prefill_speed_and_numerics() -> Result<()> {
    let Some(directory) = std::env::var_os("XINFER_NEMOTRON_PREFILL_CHECKPOINT") else {
        return Ok(());
    };
    let directory = Path::new(&directory);
    let length = std::env::var("XINFER_NEMOTRON_PREFILL_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2048);
    ensure!(length >= 2 && length % CHUNK.min(length) == 0);
    let device = Device::new_cuda(0)?;
    let began = Instant::now();
    let (model, config) = load(directory, &device)?;
    model.preallocate_mamba_cache(2)?;
    eprintln!(
        "model_load_ms={:.1}",
        began.elapsed().as_secs_f64() * 1000.0
    );
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let sample = tokenizer
        .encode(
            "A long document can mix history, engineering, mathematics, and everyday language. The model must continue each sentence from its exact preceding context. ",
            false,
        )
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    ensure!(!sample.is_empty());
    let tokens = sample
        .iter()
        .copied()
        .cycle()
        .take(length + 256)
        .collect::<Vec<_>>();

    let old_kv = cache(&config, length + 256, &device)?;
    let (old_logits, old_ms) = prefill(&model, &device, &old_kv, 0, &tokens[..length], false)?;
    eprintln!(
        "old_prefill_tokens={length} old_ms={old_ms:.3} old_ms_per_token={:.6}",
        old_ms / length as f64
    );
    let new_kv = cache(&config, length + 256, &device)?;
    let (new_logits, new_ms) = prefill(&model, &device, &new_kv, 1, &tokens[..length], true)?;
    eprintln!(
        "new_prefill_tokens={length} new_ms={new_ms:.3} new_ms_per_token={:.6}",
        new_ms / length as f64
    );
    let mut max_abs = old_logits
        .iter()
        .zip(&new_logits)
        .map(|(old, new)| (old - new).abs())
        .fold(0f32, f32::max);
    let mut top1_agree = usize::from(top1(&old_logits) == top1(&new_logits));
    if length == 2048 {
        let first_state = model.export_mamba_state_bytes(1, length as u64, [0; 32])?;
        for index in 0..256 {
            let old = forward(
                &model,
                &device,
                &old_kv,
                0,
                &tokens[length + index..length + index + 1],
                length + index,
            )?;
            let new = forward(
                &model,
                &device,
                &new_kv,
                1,
                &tokens[length + index..length + index + 1],
                length + index,
            )?;
            max_abs = max_abs.max(
                old.iter()
                    .zip(&new)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max),
            );
            top1_agree += usize::from(top1(&old) == top1(&new));
        }
        eprintln!("teacher_forced_positions=257 max_abs_logit_diff={max_abs:.8} top1_agree={top1_agree}/257");
        model.release_sequence_state(0);
        model.release_sequence_state(1);
        drop(old_kv);
        drop(new_kv);
        let repeat_kv = cache(&config, length + 256, &device)?;
        let (repeat_logits, repeat_ms) =
            prefill(&model, &device, &repeat_kv, 0, &tokens[..length], true)?;
        let repeat_state = model.export_mamba_state_bytes(0, length as u64, [0; 32])?;
        ensure!(
            new_logits == repeat_logits,
            "fast prefill logits are not bit-identical on repeat"
        );
        ensure!(
            first_state == repeat_state,
            "fast prefill recurrent state is not bit-identical on repeat"
        );
        eprintln!(
            "warm_new_prefill_ms={repeat_ms:.3} warm_new_ms_per_token={:.6}",
            repeat_ms / length as f64
        );
        eprintln!("same_schedule_repeat_logits_and_state_bit_identical=true");
    } else {
        eprintln!(
            "last_prefill_max_abs_logit_diff={max_abs:.8} last_prefill_top1_agree={top1_agree}/1"
        );
    }
    Ok(())
}
