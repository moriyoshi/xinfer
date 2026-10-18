//! Opt-in Gemma 4 E4B differential test against saved Hugging Face logits.
//! Set XINFER_GEMMA4_CHECKPOINT and XINFER_GEMMA4_REFERENCE_DIR.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
use tokenizers::Tokenizer;
use xinfer::{
    models::{
        gemma4::Gemma4ForCausalLM,
        layers::{distributed::Comm, VarBuilderX},
    },
    utils::{
        config::{Config, EngineConfig},
        gemma4_per_layer_cache_config, init_config_tokenizer,
        progress::ProgressReporter,
    },
};

const BLOCK: usize = 16;

fn engine_config(dir: &Path) -> EngineConfig {
    EngineConfig::new(
        None,
        Some(dir.display().to_string()),
        None,
        None,
        None,
        None,
        Some(1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        true,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        true,
        None,
        1,
        0,
        None,
        29500,
        false,
        None,
        None,
    )
}

fn cache(
    config: &Config,
    length: usize,
    compact: bool,
    device: &Device,
) -> Result<Vec<(Tensor, Tensor)>> {
    let shapes = gemma4_per_layer_cache_config(config).expect("Gemma 4 per-layer KV shapes");
    let first_shared = config.num_hidden_layers - 18;
    let layers = if compact { first_shared } else { shapes.len() };
    shapes[..layers]
        .iter()
        .map(|&(heads, dim)| {
            let shape = (length.div_ceil(BLOCK), BLOCK, heads, dim);
            Ok((
                Tensor::zeros(shape, DType::BF16, device)?,
                Tensor::zeros(shape, DType::BF16, device)?,
            ))
        })
        .collect()
}

fn forward(
    model: &Gemma4ForCausalLM,
    device: &Device,
    kv: &Vec<(Tensor, Tensor)>,
    tokens: &[u32],
    start: usize,
    prefill: bool,
) -> Result<Vec<f32>> {
    let end = start + tokens.len();
    let _guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
    let positions: Vec<i64> = (start..end).map(|i| i as i64).collect();
    let block_ids: Vec<u32> = (0..end.div_ceil(BLOCK) as u32).collect();
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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn load_model(dir: &Path, device: &Device) -> Result<(Gemma4ForCausalLM, Config)> {
    let (paths, is_gguf, config, _, _, _) = init_config_tokenizer(&engine_config(dir))?;
    ensure!(!is_gguf && config.num_hidden_layers == 42);
    let vb = VarBuilderX::new(&paths, false, DType::BF16, device)?;
    let model = Gemma4ForCausalLM::new(
        &vb,
        Rc::new(Comm::default()),
        &config,
        DType::BF16,
        false,
        device,
        Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
    )?;
    Ok((model, config))
}

#[test]
#[ignore = "requires a real NVFP4 checkpoint and an idle CUDA GPU"]
fn bench_real_e4b_nvfp4_single_sequence() -> Result<()> {
    let model_dir = std::env::var_os("XINFER_GEMMA4_NVFP4_CHECKPOINT")
        .expect("XINFER_GEMMA4_NVFP4_CHECKPOINT is required");
    let bench_dir =
        std::env::var_os("XINFER_GEMMA4_BENCH_DIR").expect("XINFER_GEMMA4_BENCH_DIR is required");
    let device = Device::new_cuda(0)?;
    let (model, config) = load_model(Path::new(&model_dir), &device)?;
    ensure!(
        config
            .quantization_config
            .as_ref()
            .is_some_and(|quant| quant.quant_method == "nvfp4"),
        "expected an NVFP4 checkpoint"
    );
    let sizes = [40usize, 2048, 8192];
    let mut prompts = Vec::new();
    for size in sizes {
        let path = Path::new(&bench_dir).join(format!("tokens-{size}.json"));
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(tokens.len() == size, "wrong token count for {size}");
        prompts.push(tokens);
    }
    let warm_cache = cache(&config, 64, true, &device)?;
    let warm = forward(&model, &device, &warm_cache, &prompts[0], 0, true)?;
    forward(
        &model,
        &device,
        &warm_cache,
        &[argmax(&warm) as u32],
        prompts[0].len(),
        false,
    )?;
    drop(warm_cache);

    for tokens in prompts {
        for sample in 0..3 {
            let kv = cache(&config, tokens.len() + 64, true, &device)?;
            let started = Instant::now();
            let logits = forward(&model, &device, &kv, &tokens, 0, true)?;
            let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
            let first_token = argmax(&logits);
            let mut next = first_token as u32;
            let started = Instant::now();
            for step in 0..48 {
                let logits = forward(&model, &device, &kv, &[next], tokens.len() + step, false)?;
                next = argmax(&logits) as u32;
            }
            let decode_ms_per_step = started.elapsed().as_secs_f64() * 1000.0 / 48.0;
            eprintln!(
                "gemma4_nvfp4_bench tokens={} sample={} prefill_ms={:.3} decode_ms_per_step={:.3} first_token={}",
                tokens.len(),
                sample,
                prefill_ms,
                decode_ms_per_step,
                first_token
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_fused_gelu_matches_baseline() -> Result<()> {
    real_e4b_nvfp4_fast_modes_match_baseline(true, false, false)
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_fused_gelu_and_rmsnorm_match_baseline() -> Result<()> {
    real_e4b_nvfp4_fast_modes_match_baseline(true, true, false)
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_default_fusions_match_opt_out() -> Result<()> {
    real_e4b_nvfp4_fast_modes_match_baseline(true, true, true)
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_fused_qk_rope_matches_baseline() -> Result<()> {
    real_e4b_nvfp4_single_fusion_matches_baseline("XINFER_GEMMA4_FUSED_QK_ROPE", None)
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_fused_layer_scale_matches_baseline() -> Result<()> {
    real_e4b_nvfp4_single_fusion_matches_baseline("XINFER_GEMMA4_FUSED_LAYER_SCALE", None)
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_resident_scales_matches_baseline() -> Result<()> {
    real_e4b_nvfp4_single_fusion_matches_baseline("XINFER_NVFP4_RESIDENT_SCALES", Some("1"))
}

#[test]
#[ignore = "requires the real NVFP4 checkpoint and two model instances on CUDA"]
fn real_e4b_nvfp4_gated_gelu_fp4_matches_baseline() -> Result<()> {
    real_e4b_nvfp4_single_fusion_matches_baseline("XINFER_GEMMA4_FUSED_GELU_FP4", Some("1"))
}

#[test]
fn nvfp4_resident_scales_reject_malformed_tensors() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::zeros((1, 16), DType::F16, &device)?;
    let weight = Tensor::zeros((1, 8), DType::U8, &device)?;
    let scales = Tensor::zeros((1, 1), DType::U8, &device)?;
    for invalid in [
        Tensor::zeros((1,), DType::F32, &device)?,
        Tensor::zeros((2,), DType::F16, &device)?,
    ] {
        let result = attention_rs::nvfp4_linear::nvfp4_matmul_with_device_scales(
            &input,
            &weight,
            &scales,
            1.0,
            1.0,
            None,
            true,
            None,
            None,
            Some(&invalid),
        );
        ensure!(result.is_err(), "malformed device scales were accepted");
    }
    Ok(())
}

fn real_e4b_nvfp4_single_fusion_matches_baseline(
    setting: &str,
    enabled_value: Option<&str>,
) -> Result<()> {
    let model_dir = std::env::var_os("XINFER_GEMMA4_NVFP4_CHECKPOINT")
        .expect("XINFER_GEMMA4_NVFP4_CHECKPOINT is required");
    let bench_dir =
        std::env::var_os("XINFER_GEMMA4_BENCH_DIR").expect("XINFER_GEMMA4_BENCH_DIR is required");
    let device = Device::new_cuda(0)?;
    let previous = std::env::var_os(setting);
    std::env::set_var(setting, "0");
    let (baseline, config) = load_model(Path::new(&model_dir), &device)?;
    if let Some(value) = enabled_value {
        std::env::set_var(setting, value);
    } else {
        std::env::remove_var(setting);
    }
    let (fused, fused_config) = load_model(Path::new(&model_dir), &device)?;
    if let Some(value) = previous {
        std::env::set_var(setting, value);
    } else {
        std::env::remove_var(setting);
    }
    ensure!(config.hidden_size == fused_config.hidden_size);
    for size in [40usize, 2048, 8192] {
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(
            Path::new(&bench_dir).join(format!("tokens-{size}.json")),
        )?)?;
        let base_cache = cache(&config, size + 64, true, &device)?;
        let fused_cache = cache(&config, size + 64, true, &device)?;
        let mut base_logits = forward(&baseline, &device, &base_cache, &tokens, 0, true)?;
        let mut fused_logits = forward(&fused, &device, &fused_cache, &tokens, 0, true)?;
        let prefix_max_abs = max_abs(&base_logits, &fused_logits);
        let mut continuation_max_abs = 0.0f32;
        let mut top1_equal = usize::from(argmax(&base_logits) == argmax(&fused_logits));
        let mut next = argmax(&base_logits) as u32;
        for step in 0..48 {
            base_logits = forward(&baseline, &device, &base_cache, &[next], size + step, false)?;
            fused_logits = forward(&fused, &device, &fused_cache, &[next], size + step, false)?;
            continuation_max_abs = continuation_max_abs.max(max_abs(&base_logits, &fused_logits));
            top1_equal += usize::from(argmax(&base_logits) == argmax(&fused_logits));
            next = argmax(&base_logits) as u32;
        }
        eprintln!(
            "gemma4_fusion_parity setting={setting} tokens={size} prefix_max_abs={prefix_max_abs} continuation_max_abs={continuation_max_abs} top1={top1_equal}/49"
        );
        ensure!(
            prefix_max_abs == 0.0 && continuation_max_abs == 0.0,
            "{setting} changed logits at {size}"
        );
        ensure!(
            top1_equal == 49,
            "{setting} changed a top-1 choice at {size}"
        );
    }
    Ok(())
}

fn real_e4b_nvfp4_fast_modes_match_baseline(
    fused_gelu: bool,
    fused_rms_norm: bool,
    use_default: bool,
) -> Result<()> {
    let model_dir = std::env::var_os("XINFER_GEMMA4_NVFP4_CHECKPOINT")
        .expect("XINFER_GEMMA4_NVFP4_CHECKPOINT is required");
    let bench_dir =
        std::env::var_os("XINFER_GEMMA4_BENCH_DIR").expect("XINFER_GEMMA4_BENCH_DIR is required");
    let device = Device::new_cuda(0)?;

    // This ignored test is run by name in its own process. Select the mode at
    // construction, then restore the process environment before inference.
    let previous_gelu = std::env::var_os("XINFER_GEMMA4_FUSED_GELU");
    let previous_rms = std::env::var_os("XINFER_GEMMA4_FUSED_RMSNORM");
    std::env::set_var("XINFER_GEMMA4_FUSED_GELU", "0");
    std::env::set_var("XINFER_GEMMA4_FUSED_RMSNORM", "0");
    let (baseline, config) = load_model(Path::new(&model_dir), &device)?;
    if use_default {
        std::env::remove_var("XINFER_GEMMA4_FUSED_GELU");
        std::env::remove_var("XINFER_GEMMA4_FUSED_RMSNORM");
    } else {
        std::env::set_var(
            "XINFER_GEMMA4_FUSED_GELU",
            if fused_gelu { "1" } else { "0" },
        );
        std::env::set_var(
            "XINFER_GEMMA4_FUSED_RMSNORM",
            if fused_rms_norm { "1" } else { "0" },
        );
    }
    let (fused, fused_config) = load_model(Path::new(&model_dir), &device)?;
    if let Some(value) = previous_gelu {
        std::env::set_var("XINFER_GEMMA4_FUSED_GELU", value);
    } else {
        std::env::remove_var("XINFER_GEMMA4_FUSED_GELU");
    }
    if let Some(value) = previous_rms {
        std::env::set_var("XINFER_GEMMA4_FUSED_RMSNORM", value);
    } else {
        std::env::remove_var("XINFER_GEMMA4_FUSED_RMSNORM");
    }
    ensure!(config.hidden_size == fused_config.hidden_size);

    for size in [40usize, 2048, 8192] {
        let path = Path::new(&bench_dir).join(format!("tokens-{size}.json"));
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
        let base_cache = cache(&config, size + 64, true, &device)?;
        let fused_cache = cache(&config, size + 64, true, &device)?;
        let mut base_logits = forward(&baseline, &device, &base_cache, &tokens, 0, true)?;
        let mut fused_logits = forward(&fused, &device, &fused_cache, &tokens, 0, true)?;
        let prefix_max_abs = max_abs(&base_logits, &fused_logits);
        let mut continuation_max_abs = 0.0f32;
        let mut top1_equal = usize::from(argmax(&base_logits) == argmax(&fused_logits));
        let mut next = argmax(&base_logits) as u32;
        for step in 0..48 {
            base_logits = forward(&baseline, &device, &base_cache, &[next], size + step, false)?;
            fused_logits = forward(&fused, &device, &fused_cache, &[next], size + step, false)?;
            continuation_max_abs = continuation_max_abs.max(max_abs(&base_logits, &fused_logits));
            top1_equal += usize::from(argmax(&base_logits) == argmax(&fused_logits));
            next = argmax(&base_logits) as u32;
        }
        eprintln!(
            "gemma4_fast_mode_parity gelu={fused_gelu} rms={fused_rms_norm} default={use_default} tokens={size} prefix_max_abs={prefix_max_abs} continuation_max_abs={continuation_max_abs} top1={top1_equal}/49"
        );
        ensure!(
            top1_equal == 49,
            "fast modes changed a top-1 choice at {size}"
        );
    }
    Ok(())
}

#[test]
fn real_e4b_nvfp4_load_prefill_and_continuation() -> Result<()> {
    let Some(quant_dir) = std::env::var_os("XINFER_GEMMA4_NVFP4_CHECKPOINT") else {
        return Ok(());
    };
    let bf16_dir = std::env::var_os("XINFER_GEMMA4_CHECKPOINT")
        .expect("XINFER_GEMMA4_CHECKPOINT required with NVFP4 checkpoint");
    let reference = std::env::var_os("XINFER_GEMMA4_REFERENCE_DIR")
        .expect("XINFER_GEMMA4_REFERENCE_DIR required with NVFP4 checkpoint");
    let reference = Path::new(&reference);
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(reference.join("tokens.json"))?)?;
    let reference_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("hf.json"))?)?;
    let teacher: Vec<u32> = serde_json::from_value(reference_json["greedy"].clone())?;
    let device = Device::new_cuda(0)?;
    let (quant, quant_config) = load_model(Path::new(&quant_dir), &device)?;
    ensure!(
        quant_config
            .quantization_config
            .as_ref()
            .is_some_and(|config| config.quant_method == "nvfp4"),
        "expected native NVFP4 checkpoint"
    );
    let (bf16, bf16_config) = load_model(Path::new(&bf16_dir), &device)?;
    ensure!(bf16_config.quantization_config.is_none());
    let quant_cache = cache(&quant_config, 768, true, &device)?;
    let bf16_cache = cache(&bf16_config, 768, true, &device)?;
    let quant_start = Instant::now();
    let quant_logits = forward(&quant, &device, &quant_cache, &tokens, 0, true)?;
    eprintln!(
        "gemma4_nvfp4_prefill_ms={:.3}",
        quant_start.elapsed().as_secs_f64() * 1000.0
    );
    let bf16_logits = forward(&bf16, &device, &bf16_cache, &tokens, 0, true)?;
    ensure!(quant_logits.len() == bf16_logits.len());
    ensure!(quant_logits.iter().all(|value| value.is_finite()));
    eprintln!(
        "gemma4_nvfp4_prefix_max_abs={} top1={}/{}",
        max_abs(&quant_logits, &bf16_logits),
        argmax(&quant_logits),
        argmax(&bf16_logits)
    );
    ensure!(
        argmax(&quant_logits) == argmax(&bf16_logits),
        "NVFP4 prefix top choice differs from BF16"
    );

    let quant_repeat_cache = cache(&quant_config, 768, true, &device)?;
    let quant_repeat = forward(&quant, &device, &quant_repeat_cache, &tokens, 0, true)?;
    ensure!(
        quant_logits == quant_repeat,
        "NVFP4 prefill is not repeatable"
    );

    let mut choices_agree = 0usize;
    let mut different_choices = Vec::new();
    let mut largest_logit_diff = 0.0f32;
    let mut quant_decode_ms = 0.0f64;
    let mut bf16_decode_ms = 0.0f64;
    let mut replay_logits = Vec::new();
    for (step, &token) in teacher.iter().enumerate() {
        let started = Instant::now();
        let quant_next = forward(
            &quant,
            &device,
            &quant_cache,
            &[token],
            tokens.len() + step,
            false,
        )?;
        quant_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
        if step < 4 {
            replay_logits.push(quant_next.clone());
        }
        let started = Instant::now();
        let bf16_next = forward(
            &bf16,
            &device,
            &bf16_cache,
            &[token],
            tokens.len() + step,
            false,
        )?;
        bf16_decode_ms += started.elapsed().as_secs_f64() * 1000.0;
        ensure!(quant_next.iter().all(|value| value.is_finite()));
        if argmax(&quant_next) == argmax(&bf16_next) {
            choices_agree += 1;
        } else {
            different_choices.push(step);
        }
        largest_logit_diff = largest_logit_diff.max(max_abs(&quant_next, &bf16_next));
    }
    eprintln!(
        "gemma4_nvfp4_teacher_forced_top1={choices_agree}/{} differing_positions={different_choices:?} max_abs={largest_logit_diff} nvfp4_ms_per_token={:.3} bf16_ms_per_token={:.3}",
        teacher.len(),
        quant_decode_ms / teacher.len() as f64,
        bf16_decode_ms / teacher.len() as f64,
    );
    ensure!(
        choices_agree >= 40,
        "NVFP4 disagrees with BF16 at too many teacher-forced positions: {choices_agree}/{}",
        teacher.len()
    );
    for (step, &token) in teacher.iter().take(replay_logits.len()).enumerate() {
        let replay = forward(
            &quant,
            &device,
            &quant_repeat_cache,
            &[token],
            tokens.len() + step,
            false,
        )?;
        ensure!(
            replay == replay_logits[step],
            "NVFP4 decode changed on repeated schedule at step {step}"
        );
    }

    let quant_greedy_cache = cache(&quant_config, 768, true, &device)?;
    let bf16_greedy_cache = cache(&bf16_config, 768, true, &device)?;
    let mut quant_next = argmax(&forward(
        &quant,
        &device,
        &quant_greedy_cache,
        &tokens,
        0,
        true,
    )?) as u32;
    let mut bf16_next = argmax(&forward(
        &bf16,
        &device,
        &bf16_greedy_cache,
        &tokens,
        0,
        true,
    )?) as u32;
    let mut quant_greedy = Vec::new();
    let mut bf16_greedy = Vec::new();
    for step in 0..teacher.len() {
        quant_greedy.push(quant_next);
        bf16_greedy.push(bf16_next);
        quant_next = argmax(&forward(
            &quant,
            &device,
            &quant_greedy_cache,
            &[quant_next],
            tokens.len() + step,
            false,
        )?) as u32;
        bf16_next = argmax(&forward(
            &bf16,
            &device,
            &bf16_greedy_cache,
            &[bf16_next],
            tokens.len() + step,
            false,
        )?) as u32;
    }
    let matching_prefix = quant_greedy
        .iter()
        .zip(&bf16_greedy)
        .take_while(|(left, right)| left == right)
        .count();
    let tokenizer = Tokenizer::from_file(Path::new(&bf16_dir).join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    eprintln!(
        "gemma4_nvfp4_greedy_matching_prefix={matching_prefix}/{} nvfp4_text={} bf16_text={}",
        teacher.len(),
        tokenizer
            .decode(&quant_greedy, true)
            .map_err(anyhow::Error::msg)?,
        tokenizer
            .decode(&bf16_greedy, true)
            .map_err(anyhow::Error::msg)?,
    );
    Ok(())
}

#[test]
fn real_e4b_shared_kv_matches_hf_and_continuations() -> Result<()> {
    let Some(dir) = std::env::var_os("XINFER_GEMMA4_CHECKPOINT") else {
        return Ok(());
    };
    let reference = std::env::var_os("XINFER_GEMMA4_REFERENCE_DIR")
        .expect("XINFER_GEMMA4_REFERENCE_DIR required with checkpoint");
    let dir = Path::new(&dir);
    let reference = Path::new(&reference);
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(reference.join("tokens.json"))?)?;
    let hf_bytes = std::fs::read(reference.join("hf.f32"))?;
    let hf = hf_bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let (paths, is_gguf, config, _, _, _) = init_config_tokenizer(&engine_config(dir))?;
    ensure!(
        !is_gguf && config.num_hidden_layers == 42,
        "expected BF16 Gemma 4 E4B"
    );
    let device = Device::new_cuda(0)?;
    let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
    let model = Gemma4ForCausalLM::new(
        &vb,
        Rc::new(Comm::default()),
        &config,
        DType::BF16,
        false,
        &device,
        Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
    )?;
    ensure!(model.num_kv_cache_layers() == 24);
    ensure!(model.kv_cache_source_layers().len() == 42);
    ensure!(model.kv_cache_source_layers()[24] == 22);
    ensure!(model.kv_cache_source_layers()[29] == 23);
    let capacity = 768;
    let full_cache = cache(&config, capacity, false, &device)?;
    let full = forward(&model, &device, &full_cache, &tokens, 0, true)?;
    let repeat_cache = cache(&config, capacity, true, &device)?;
    let repeat = forward(&model, &device, &repeat_cache, &tokens, 0, true)?;
    ensure!(full == repeat, "repeated prefill logits differ");
    let hf_max_abs = full
        .iter()
        .zip(hf.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    ensure!(
        hf.len() == full.len(),
        "HF and Xinfer vocabulary sizes differ"
    );
    eprintln!(
        "gemma4_hf_max_abs={hf_max_abs} hf_top1={} xinfer_top1={}",
        argmax(&hf),
        argmax(&full)
    );
    ensure!(
        argmax(&full) == argmax(&hf),
        "Gemma 4 top-1 differs from HF"
    );
    ensure!(
        hf_max_abs < 1.0,
        "Gemma 4 logits differ substantially from HF"
    );
    let compact_cache = cache(&config, capacity, true, &device)?;
    let compact = forward(&model, &device, &compact_cache, &tokens, 0, true)?;
    ensure!(full == compact, "24-pair and 42-pair cache logits differ");
    let split_cache = cache(&config, capacity, true, &device)?;
    forward(&model, &device, &split_cache, &tokens[..32], 0, true)?;
    let split = forward(&model, &device, &split_cache, &tokens[32..], 32, true)?;
    eprintln!(
        "gemma4_full_vs_chunk_max_abs={} top1={}/{}",
        max_abs(&full, &split),
        argmax(&full),
        argmax(&split)
    );
    ensure!(
        argmax(&full) == argmax(&split),
        "continued prefill top-1 differs"
    );
    let split_repeat_cache = cache(&config, capacity, true, &device)?;
    forward(&model, &device, &split_repeat_cache, &tokens[..32], 0, true)?;
    let split_repeat = forward(
        &model,
        &device,
        &split_repeat_cache,
        &tokens[32..],
        32,
        true,
    )?;
    ensure!(split == split_repeat, "continued prefill is not repeatable");
    let decode_cache = cache(&config, capacity, true, &device)?;
    forward(&model, &device, &decode_cache, &tokens[..32], 0, true)?;
    let mut decoded = Vec::new();
    for (offset, &token) in tokens[32..].iter().enumerate() {
        decoded = forward(&model, &device, &decode_cache, &[token], 32 + offset, false)?;
    }
    eprintln!(
        "gemma4_full_vs_decode_max_abs={} top1={}/{}",
        max_abs(&full, &decoded),
        argmax(&full),
        argmax(&decoded)
    );
    ensure!(
        argmax(&full) == argmax(&decoded),
        "single-token continuation top-1 differs"
    );
    let decode_repeat_cache = cache(&config, capacity, true, &device)?;
    forward(
        &model,
        &device,
        &decode_repeat_cache,
        &tokens[..32],
        0,
        true,
    )?;
    let mut decoded_repeat = Vec::new();
    for (offset, &token) in tokens[32..].iter().enumerate() {
        decoded_repeat = forward(
            &model,
            &device,
            &decode_repeat_cache,
            &[token],
            32 + offset,
            false,
        )?;
    }
    ensure!(
        decoded == decoded_repeat,
        "single-token decode is not repeatable"
    );

    let reference_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("hf.json"))?)?;
    let expected_greedy: Vec<u32> = serde_json::from_value(reference_json["greedy"].clone())?;
    let mut next = argmax(&full) as u32;
    let mut greedy = Vec::new();
    #[cfg(feature = "nvtx")]
    nvtx::range_push!("gemma4_greedy_decode");
    let decode_started = Instant::now();
    for (step, &_expected) in expected_greedy.iter().enumerate() {
        greedy.push(next);
        next = argmax(&forward(
            &model,
            &device,
            &full_cache,
            &[next],
            tokens.len() + step,
            false,
        )?) as u32;
    }
    eprintln!(
        "gemma4_greedy_decode_ms_per_token={:.3}",
        decode_started.elapsed().as_secs_f64() * 1000.0 / expected_greedy.len() as f64
    );
    #[cfg(feature = "nvtx")]
    nvtx::range_pop!();
    let greedy_matches = greedy
        .iter()
        .zip(&expected_greedy)
        .filter(|(a, b)| a == b)
        .count();
    let first_difference = greedy
        .iter()
        .zip(&expected_greedy)
        .position(|(a, b)| a != b);
    eprintln!(
        "gemma4_hf_greedy_match={}/{} first_difference={first_difference:?}",
        greedy_matches,
        expected_greedy.len(),
    );
    ensure!(first_difference.unwrap_or(greedy.len()) >= 12);
    ensure!(greedy_matches >= expected_greedy.len() / 2);
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    eprintln!(
        "gemma4_greedy_text={}",
        tokenizer
            .decode(&greedy, true)
            .map_err(anyhow::Error::msg)?
    );

    let long_tokens = tokens.iter().copied().cycle().take(640).collect::<Vec<_>>();
    let long_full_cache = cache(&config, capacity, true, &device)?;
    let long_full = forward(&model, &device, &long_full_cache, &long_tokens, 0, true)?;
    let long_split_cache = cache(&config, capacity, true, &device)?;
    forward(
        &model,
        &device,
        &long_split_cache,
        &long_tokens[..520],
        0,
        true,
    )?;
    let long_split = forward(
        &model,
        &device,
        &long_split_cache,
        &long_tokens[520..],
        520,
        true,
    )?;
    eprintln!(
        "gemma4_long_640_full_split_max_abs={} top1={}/{}",
        max_abs(&long_full, &long_split),
        argmax(&long_full),
        argmax(&long_split)
    );
    ensure!(
        argmax(&long_full) == argmax(&long_split),
        "640-token top-1 differs past sliding window"
    );
    if let Ok(hf_long_bytes) = std::fs::read(reference.join("hf_long.f32")) {
        ensure!(hf_long_bytes.len() == long_full.len() * 4);
        let hf_long = hf_long_bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let difference = max_abs(&long_full, &hf_long);
        eprintln!(
            "gemma4_long_640_hf_max_abs={difference} top1={}/{}",
            argmax(&long_full),
            argmax(&hf_long)
        );
        ensure!(argmax(&long_full) == argmax(&hf_long));
        ensure!(difference < 3.0);
    }
    if std::env::var_os("XINFER_GEMMA4_BENCH_LONG_DECODE").is_some() {
        let mut next = argmax(&long_full) as u32;
        let mut generated = Vec::new();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("gemma4_long_decode");
        let started = Instant::now();
        for step in 0..64 {
            generated.push(next);
            next = argmax(&forward(
                &model,
                &device,
                &long_full_cache,
                &[next],
                640 + step,
                false,
            )?) as u32;
        }
        eprintln!(
            "gemma4_long_decode_ms_per_token={:.3} tokens={generated:?}",
            started.elapsed().as_secs_f64() * 1000.0 / generated.len() as f64
        );
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
    }
    Ok(())
}
