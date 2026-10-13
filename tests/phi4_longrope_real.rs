//! Opt-in Phi-4-mini LongRoPE checkpoint test. Set XINFER_PHI4_CHECKPOINT.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use std::{path::Path, rc::Rc, sync::Arc};
use tokenizers::Tokenizer;
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        phi4::Phi4ForCausalLM,
    },
    utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
};

const BLOCK: usize = 16;
type Cache = Vec<(Tensor, Tensor)>;

fn load(directory: &Path, device: &Device) -> Result<(Phi4ForCausalLM, Config)> {
    let config: Config = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let mut filenames = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    filenames.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
    filenames.sort();
    ensure!(
        !filenames.is_empty(),
        "checkpoint has no safetensors shards"
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
    let model = Phi4ForCausalLM::new(
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

fn cache(config: &Config, capacity: usize, device: &Device) -> Result<Cache> {
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);
    let shape = (
        capacity.div_ceil(BLOCK),
        BLOCK,
        config.num_key_value_heads,
        head_dim,
    );
    (0..config.num_hidden_layers)
        .map(|_| {
            Ok((
                Tensor::zeros(shape, DType::BF16, device)?,
                Tensor::zeros(shape, DType::BF16, device)?,
            ))
        })
        .collect()
}

fn run(
    model: &Phi4ForCausalLM,
    device: &Device,
    kv: &Cache,
    id: usize,
    tokens: &[u32],
    start: usize,
) -> Result<Vec<f32>> {
    let prefill = tokens.len() > 1 || start == 0;
    let _guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
    let end = start + tokens.len();
    let positions: Vec<i64> = (start..end).map(|position| position as i64).collect();
    let blocks: Vec<u32> = (0..end.div_ceil(BLOCK) as u32).collect();
    let metadata = InputMetadata {
        is_prefill: prefill,
        is_mla: false,
        sequence_ids: Some(vec![id]),
        mamba_slot_mapping: None,
        slot_mapping: Tensor::from_vec(positions.clone(), tokens.len(), device)?,
        block_tables: Some(Tensor::from_vec(blocks.clone(), (1, blocks.len()), device)?),
        block_tables_host: Some(vec![blocks]),
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

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn top1(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[test]
fn real_phi_longrope_full_chunked_and_restored() -> Result<()> {
    let Some(directory) = std::env::var_os("XINFER_PHI4_CHECKPOINT") else {
        return Ok(());
    };
    let directory = Path::new(&directory);
    let device = Device::new_cuda(0)?;
    let (model, config) = load(directory, &device)?;
    ensure!(
        config.original_max_position_embeddings == Some(4096),
        "expected original window 4096"
    );
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let sample = tokenizer.encode("The history of computing includes algorithms, hardware, operating systems, and mathematical ideas. ", false)
        .map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let tokens = sample.into_iter().cycle().take(7001).collect::<Vec<_>>();

    let short = cache(&config, tokens.len(), &device)?;
    let short_logits = run(&model, &device, &short, 1, &tokens[..256], 0)?;
    ensure!(
        short_logits.iter().all(|value| value.is_finite()),
        "short logits are nonfinite"
    );
    let _ = run(&model, &device, &short, 1, &tokens[256..4096], 256)?;
    let boundary = run(&model, &device, &short, 1, &tokens[4096..4112], 4096);
    ensure!(
        boundary.is_err(),
        "short KV crossed the LongRoPE boundary silently"
    );

    let full = cache(&config, tokens.len(), &device)?;
    let _ = run(&model, &device, &full, 2, &tokens[..4112], 0)?;
    let full_next = run(&model, &device, &full, 2, &tokens[4112..4113], 4112)?;

    let chunks = cache(&config, tokens.len(), &device)?;
    model.prepare_long_rope_cache(3, tokens.len())?;
    let _ = run(&model, &device, &chunks, 3, &tokens[..4096], 0)?;
    let _ = run(&model, &device, &chunks, 3, &tokens[4096..4112], 4096)?;
    let chunk_next = run(&model, &device, &chunks, 3, &tokens[4112..4113], 4112)?;
    let diff = max_abs(&full_next, &chunk_next);
    eprintln!("Phi full versus prepared-long chunked next-logit max_abs={diff}");
    ensure!(diff < 0.05, "full/chunked LongRoPE logits diverged");

    let binding = model.export_rope_cache_binding(3)?;
    ensure!(
        binding.prefix_len == 4113 && binding.uses_long_rope,
        "unexpected binding"
    );
    let wire = serde_json::to_vec(&binding)?;
    let restored_binding = serde_json::from_slice(&wire)?;
    let restored = chunks
        .iter()
        .map(|(k, v)| {
            Ok((
                k.to_device(&Device::Cpu)?.to_device(&device)?,
                v.to_device(&Device::Cpu)?.to_device(&device)?,
            ))
        })
        .collect::<Result<Cache>>()?;
    model.bind_restored_rope_cache(4, &restored_binding)?;
    let hot = run(&model, &device, &chunks, 3, &tokens[4113..4114], 4113)?;
    let cold = run(&model, &device, &restored, 4, &tokens[4113..4114], 4113)?;
    let restore_diff = max_abs(&hot, &cold);
    eprintln!("Phi restored versus hot next-logit max_abs={restore_diff}");
    ensure!(restore_diff == 0.0, "restored Phi cache changed logits");
    let _ = run(&model, &device, &full, 2, &tokens[4113..7000], 4113)?;
    let full_7k = run(&model, &device, &full, 2, &tokens[7000..7001], 7000)?;
    let _ = run(&model, &device, &chunks, 3, &tokens[4114..7000], 4114)?;
    let chunked_7k = run(&model, &device, &chunks, 3, &tokens[7000..7001], 7000)?;
    let fresh_7k_cache = cache(&config, tokens.len(), &device)?;
    let _ = run(&model, &device, &fresh_7k_cache, 6, &tokens[..7000], 0)?;
    let fresh_7k = run(
        &model,
        &device,
        &fresh_7k_cache,
        6,
        &tokens[7000..7001],
        7000,
    )?;
    ensure!(
        fresh_7k.iter().all(|value| value.is_finite()),
        "7K long logits are nonfinite"
    );
    let chunk_diff = max_abs(&fresh_7k, &chunked_7k);
    let continued_diff = max_abs(&fresh_7k, &full_7k);
    eprintln!("Phi 7K fresh versus chunked max_abs={chunk_diff}; fresh versus continued max_abs={continued_diff}; top1 fresh/chunked/continued={}/{}/{}",
        top1(&fresh_7k), top1(&chunked_7k), top1(&full_7k));
    // Full and continued BF16 prefill can differ numerically. Report that
    // drift separately; the restored-cache oracle above remains exact.
    ensure!(
        top1(&fresh_7k) == top1(&chunked_7k) && top1(&fresh_7k) == top1(&full_7k),
        "Phi 7K full/chunked top choice changed"
    );

    let mut corrupt = binding;
    corrupt.rope_fingerprint[0] ^= 1;
    ensure!(
        model.bind_restored_rope_cache(5, &corrupt).is_err(),
        "mismatched RoPE binding accepted"
    );
    Ok(())
}
