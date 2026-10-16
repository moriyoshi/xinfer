//! Opt-in Gemma 4 E4B differential test against saved Hugging Face logits.
//! Set XINFER_GEMMA4_CHECKPOINT and XINFER_GEMMA4_REFERENCE_DIR.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::{path::Path, rc::Rc, sync::Arc};
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
    Ok(())
}
