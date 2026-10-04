//! Opt-in real-checkpoint transfer between independent Nemotron-H instances.
//! Set XINFER_NEMOTRON_H_CHECKPOINT to the pinned local checkpoint directory.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Context, Result};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use parking_lot::RwLock;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
use tokenizers::Tokenizer;
use xinfer::{
    models::{
        layers::{distributed::Comm, VarBuilderX},
        nemotron_h::{NemotronHForCausalLM, NemotronMambaSnapshot},
    },
    utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
};

const BLOCK: usize = 16;
type Cache = Vec<(Tensor, Tensor)>;
type AttentionImage = (Vec<usize>, Vec<bf16>);

fn export_attention(tensor: &Tensor) -> Result<AttentionImage> {
    Ok((
        tensor.dims().to_vec(),
        tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?,
    ))
}

fn import_attention(image: &AttentionImage, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        image.1.clone(),
        image.0.as_slice(),
        device,
    )?)
}

struct Model {
    device: Device,
    config: Config,
    inner: NemotronHForCausalLM,
}

impl Model {
    fn load(directory: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(directory.join("config.json"))?;
        let root: Value = serde_json::from_str(&raw)?;
        ensure!(root["architectures"][0] == "NemotronHForCausalLM");
        let mut config: Config = serde_json::from_str(&raw)?;
        config.extra_config_json = Some(raw);
        let index: Value = serde_json::from_slice(&std::fs::read(
            directory.join("model.safetensors.index.json"),
        )?)?;
        let mut filenames = index["weight_map"]
            .as_object()
            .context("missing weight map")?
            .values()
            .filter_map(Value::as_str)
            .map(|name| directory.join(name))
            .collect::<Vec<_>>();
        filenames.sort();
        filenames.dedup();
        ensure!(filenames.iter().all(|path| path.exists()), "missing shard");
        let paths = ModelPaths {
            tokenizer_filename: directory.join("tokenizer.json"),
            tokenizer_config_filename: directory.join("tokenizer_config.json"),
            config_filename: directory.join("config.json"),
            generation_config_filename: directory.join("generation_config.json"),
            filenames,
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        let start = Instant::now();
        let device = xinfer::utils::new_device(0)?;
        let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
        let inner = NemotronHForCausalLM::new(
            &vb,
            Rc::new(Comm::default()),
            &config,
            DType::BF16,
            false,
            &device,
            Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        )?;
        inner.set_mamba_prefix_cache_capacity(2);
        eprintln!("model_load_s={:.3}", start.elapsed().as_secs_f64());
        Ok(Self {
            device,
            config,
            inner,
        })
    }

    fn empty_cache(&self, capacity: usize) -> Result<Cache> {
        let raw = self.config.extra_config_json.as_ref().context("config")?;
        let root: Value = serde_json::from_str(raw)?;
        let n = root["hybrid_override_pattern"]
            .as_str()
            .context("layer pattern")?
            .bytes()
            .filter(|&kind| kind == b'*')
            .count();
        let shape = (
            capacity.div_ceil(BLOCK),
            BLOCK,
            self.config.num_key_value_heads,
            self.config.head_dim.context("head_dim")?,
        );
        (0..n)
            .map(|_| {
                Ok((
                    Tensor::zeros(shape, DType::BF16, &self.device)?,
                    Tensor::zeros(shape, DType::BF16, &self.device)?,
                ))
            })
            .collect()
    }

    fn forward(&self, tokens: &[u32], start: usize, cache: &Cache) -> Result<Vec<f32>> {
        let prefill = start == 0;
        ensure!(!tokens.is_empty());
        ensure!(prefill || tokens.len() == 1);
        let _guard = xinfer::models::layers::linear::set_linear_is_prefill(prefill);
        let end = start + tokens.len();
        let positions = (start..end).map(|i| i as i64).collect::<Vec<_>>();
        let block_ids = (0..end.div_ceil(BLOCK) as u32).collect::<Vec<_>>();
        let metadata = InputMetadata {
            is_prefill: prefill,
            is_mla: false,
            sequence_ids: Some(vec![0]),
            mamba_slot_mapping: None,
            slot_mapping: Tensor::from_vec(positions.clone(), tokens.len(), &self.device)?,
            block_tables: Some(Tensor::from_vec(
                block_ids.clone(),
                (1, block_ids.len()),
                &self.device,
            )?),
            block_tables_host: Some(vec![block_ids]),
            context_lens_host: Some(vec![end as u32]),
            context_lens: Some(Tensor::from_vec(vec![end as u32], 1, &self.device)?),
            cu_seqlens_q: if prefill {
                Some(Tensor::from_vec(
                    vec![0u32, tokens.len() as u32],
                    2,
                    &self.device,
                )?)
            } else {
                None
            },
            cu_seqlens_k: if prefill {
                Some(Tensor::from_vec(vec![0u32, end as u32], 2, &self.device)?)
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
        Ok(self
            .inner
            .forward(
                &Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?,
                &Tensor::from_vec(positions, tokens.len(), &self.device)?,
                Some(cache),
                &metadata,
                false,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
}

#[test]
fn real_checkpoint_cross_instance_continuation() -> Result<()> {
    let directory = match std::env::var_os("XINFER_NEMOTRON_H_CHECKPOINT") {
        Some(path) => std::path::PathBuf::from(path),
        None => return Ok(()),
    };
    let tokenizer =
        Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    let tokens = tokenizer.encode(
        "あなたは日本語で簡潔に答えるアシスタントです。東京の天気が晴れなら、散歩に必要なものを短く挙げてください。",
        false,
    ).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let prefix_len = 20;
    ensure!(tokens.len() >= prefix_len + 4, "prompt too short");
    let source = Model::load(&directory)?;
    let source_cache = source.empty_cache(prefix_len + 4)?;
    let prefill = source.forward(&tokens[..prefix_len], 0, &source_cache)?;
    ensure!(
        prefill.iter().all(|value| value.is_finite()),
        "non-finite prefill logits"
    );
    source.device.synchronize()?;

    let fingerprint: [u8; 32] = Sha256::digest(
        b"nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese@3979dd16634988c34cc3bd911583c51e6a731d10/bf16/tp1"
    ).into();
    let export_start = Instant::now();
    let snapshot = source
        .inner
        .export_mamba_state(0, prefix_len as u64, fingerprint)?;
    source.device.synchronize()?;
    let export_s = export_start.elapsed().as_secs_f64();
    ensure!(
        snapshot.layout.model_layer_indices.len() == 27,
        "wrong Mamba layer count"
    );
    ensure!(
        snapshot.payload.len() == 145_539_072,
        "wrong Mamba payload size"
    );
    let encode_start = Instant::now();
    let encoded = snapshot.to_bytes()?;
    let encode_s = encode_start.elapsed().as_secs_f64();
    let export_bytes_start = Instant::now();
    let direct = source
        .inner
        .export_mamba_state_bytes(0, prefix_len as u64, fingerprint)?;
    source.device.synchronize()?;
    let export_bytes_s = export_bytes_start.elapsed().as_secs_f64();
    ensure!(
        direct == encoded,
        "bytes-first export changed the v1 envelope"
    );
    let attention = source_cache
        .iter()
        .map(|(key, value)| Ok((export_attention(key)?, export_attention(value)?)))
        .collect::<Result<Vec<_>>>()?;

    // A second model owns its own weights, recurrent map, and attention cache.
    let target = Model::load(&directory)?;
    let decode_start = Instant::now();
    let restored = NemotronMambaSnapshot::from_bytes(&encoded)?;
    let decode_s = decode_start.elapsed().as_secs_f64();
    ensure!(
        restored.payload == snapshot.payload,
        "parsed payload changed"
    );
    let target_cache = attention
        .iter()
        .map(|(key, value)| {
            Ok((
                import_attention(key, &target.device)?,
                import_attention(value, &target.device)?,
            ))
        })
        .collect::<Result<Cache>>()?;
    let import_start = Instant::now();
    target
        .inner
        .import_mamba_state_bytes(0, prefix_len as u64, fingerprint, &encoded)?;
    target.device.synchronize()?;
    let import_s = import_start.elapsed().as_secs_f64();
    ensure!(
        target
            .inner
            .import_mamba_state_bytes(0, prefix_len as u64, fingerprint, &encoded)
            .is_err(),
        "duplicate import was accepted"
    );

    let mut max_abs = 0.0f32;
    for i in 0..4 {
        let token = &tokens[prefix_len + i..prefix_len + i + 1];
        let baseline = source.forward(token, prefix_len + i, &source_cache)?;
        let replay = target.forward(token, prefix_len + i, &target_cache)?;
        ensure!(baseline.len() == replay.len(), "logit length changed");
        ensure!(
            baseline.iter().all(|v| v.is_finite()) && replay.iter().all(|v| v.is_finite()),
            "non-finite continuation logits"
        );
        for (&a, &b) in baseline.iter().zip(&replay) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    target.device.synchronize()?;
    eprintln!(
        "Nemotron-H snapshot export_s={export_s:.6} encode_s={encode_s:.6} decode_s={decode_s:.6} export_bytes_s={export_bytes_s:.6} import_bytes_s={import_s:.6}"
    );
    eprintln!("Nemotron-H cross-instance continuation max_abs_logit_diff={max_abs}");
    ensure!(
        max_abs <= 1e-3,
        "cross-instance continuation changed logits"
    );
    Ok(())
}
