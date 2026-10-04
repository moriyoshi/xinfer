//! Opt-in, real-checkpoint continuation test. Set XINFER_QWEN_GDN_CHECKPOINT
//! to the Qwen3.6-27B-FP8 directory and run with `--features cuda`.

#[cfg(feature = "cuda")]
mod cuda_test {
    use attention_rs::InputMetadata;
    use candle_core::{DType, Device, Tensor};
    use parking_lot::RwLock;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{path::Path, rc::Rc, sync::Arc};
    use tokenizers::Tokenizer;
    use xinfer::{
        models::{
            layers::{distributed::Comm, gdn_state::GdnStateSnapshot, VarBuilderX},
            qwen3_5::Qwen3_5ForCausalLM,
        },
        utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
    };

    type Cache = Vec<(Tensor, Tensor)>;
    const BLOCK: usize = 16;

    struct Model {
        device: Device,
        config: Config,
        inner: Qwen3_5ForCausalLM,
    }

    impl Model {
        fn load(directory: &Path) -> anyhow::Result<Self> {
            let root: Value =
                serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
            anyhow::ensure!(
                root["architectures"][0] == "Qwen3_5ForConditionalGeneration",
                "expected a Qwen3.5/3.6 conditional-generation checkpoint"
            );
            let mut text = root["text_config"].clone();
            text["architectures"] = json!(["Qwen3_5ForCausalLM"]);
            let mut config: Config = serde_json::from_value(text.clone())?;
            config.extra_config_json = Some(text.to_string());
            if let Some(quantization) = root.get("quantization_config") {
                if !quantization.is_null() {
                    config.quantization_config =
                        Some(serde_json::from_value(quantization.clone())?);
                }
            }
            let index: Value = serde_json::from_slice(&std::fs::read(
                directory.join("model.safetensors.index.json"),
            )?)?;
            let mut filenames = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("missing weight map"))?
                .values()
                .filter_map(Value::as_str)
                .filter(|name| *name != "mtp.safetensors")
                .map(|name| directory.join(name))
                .collect::<Vec<_>>();
            filenames.sort();
            filenames.dedup();
            anyhow::ensure!(filenames.iter().all(|path| path.exists()), "missing shard");
            let paths = ModelPaths {
                tokenizer_filename: directory.join("tokenizer.json"),
                tokenizer_config_filename: directory.join("tokenizer_config.json"),
                config_filename: directory.join("config.json"),
                generation_config_filename: directory.join("generation_config.json"),
                filenames,
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            };
            let device = xinfer::utils::new_device(0)?;
            let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
            let inner = Qwen3_5ForCausalLM::new_with_prefix(
                &vb,
                Rc::new(Comm::default()),
                &config,
                DType::BF16,
                false,
                &device,
                Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
                Some("model.language_model.".into()),
            )?;
            Ok(Self {
                device,
                config,
                inner,
            })
        }

        fn empty_cache(&self, capacity: usize) -> anyhow::Result<Cache> {
            let shape = (
                capacity.div_ceil(BLOCK),
                BLOCK,
                self.config.num_key_value_heads,
                self.config
                    .head_dim
                    .ok_or_else(|| anyhow::anyhow!("head_dim"))?,
            );
            (0..self.config.num_hidden_layers / 4)
                .map(|_| {
                    Ok((
                        Tensor::zeros(shape, DType::BF16, &self.device)?,
                        Tensor::zeros(shape, DType::BF16, &self.device)?,
                    ))
                })
                .collect()
        }

        fn forward(&self, tokens: &[u32], start: usize, cache: &Cache) -> anyhow::Result<Vec<f32>> {
            let prefill = start == 0;
            anyhow::ensure!(!tokens.is_empty(), "empty token input");
            anyhow::ensure!(prefill || tokens.len() == 1, "decode one token at a time");
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
    fn qwen36_27b_cross_instance_prefill_continuation() -> anyhow::Result<()> {
        let Some(directory) = std::env::var_os("XINFER_QWEN_GDN_CHECKPOINT") else {
            eprintln!("set XINFER_QWEN_GDN_CHECKPOINT to run the 27B cross-instance test");
            return Ok(());
        };
        let directory = Path::new(&directory);
        let tokenizer =
            Tokenizer::from_file(directory.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let tokens = tokenizer
            .encode(
                "The quick brown fox jumps over the lazy dog. The next sentence continues with several more words so that decoding can compare independent model instances.",
                false,
            )
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        anyhow::ensure!(tokens.len() > 12, "unexpectedly short test prompt");
        let prefix_len = tokens.len() - 4;

        // Both instances read the same verified checkpoint files. Applications
        // must fingerprint actual weights/adapters and numerical settings.
        let mut identity = Sha256::new();
        identity.update(std::fs::read(directory.join("config.json"))?);
        identity.update(std::fs::read(
            directory.join("model.safetensors.index.json"),
        )?);
        let fingerprint: [u8; 32] = identity.finalize().into();

        let source = Model::load(directory)?;
        let source_kv = source.empty_cache(tokens.len())?;
        source.forward(&tokens[..prefix_len], 0, &source_kv)?;
        source.device.synchronize()?;
        let exported = source
            .inner
            .export_gdn_state(0, prefix_len as u64, fingerprint)?;
        anyhow::ensure!(exported.layout.model_layer_indices.len() == 48);
        anyhow::ensure!(exported.payload.len() == 156_893_184);
        let wire = exported.to_bytes()?;
        let decoded = GdnStateSnapshot::from_bytes(&wire)?;

        // KV crosses host memory too, so no device tensor is shared with B.
        let host_kv = source_kv
            .iter()
            .map(|(key, value)| Ok((key.to_device(&Device::Cpu)?, value.to_device(&Device::Cpu)?)))
            .collect::<candle_core::Result<Cache>>()?;
        let target = Model::load(directory)?;
        let target_kv = host_kv
            .iter()
            .map(|(key, value)| {
                Ok((
                    key.to_device(&target.device)?,
                    value.to_device(&target.device)?,
                ))
            })
            .collect::<candle_core::Result<Cache>>()?;
        target
            .inner
            .import_gdn_state(0, prefix_len as u64, fingerprint, &decoded)?;

        let mut max_abs_logit_diff = 0.0f32;
        for i in 0..4 {
            let position = prefix_len + i;
            let a = source.forward(&tokens[position..position + 1], position, &source_kv)?;
            let b = target.forward(&tokens[position..position + 1], position, &target_kv)?;
            anyhow::ensure!(a.len() == b.len(), "logit vector length mismatch");
            for (a, b) in a.iter().zip(&b) {
                max_abs_logit_diff = max_abs_logit_diff.max((a - b).abs());
            }
        }
        eprintln!(
            "cross-instance Qwen3.6 GDN+KV transfer: {} bytes, max logit diff {max_abs_logit_diff}",
            wire.len()
        );
        anyhow::ensure!(
            max_abs_logit_diff <= 1e-3,
            "cross-instance continuation changed logits by {max_abs_logit_diff}"
        );
        Ok(())
    }
}
