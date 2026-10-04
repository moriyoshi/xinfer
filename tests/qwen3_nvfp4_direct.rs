//! Opt-in smoke test for a direct, raw-Config Qwen3 NVFP4 construction.
//! Set XINFER_QWEN3_NVFP4_CHECKPOINT to the llmat checkpoint directory.

#[cfg(feature = "cuda")]
mod cuda_test {
    use attention_rs::InputMetadata;
    use candle_core::{DType, Tensor};
    use parking_lot::RwLock;
    use std::{path::Path, rc::Rc, sync::Arc};
    use xinfer::{
        models::{
            layers::{distributed::Comm, VarBuilderX},
            qwen3::Qwen3ForCausalLM,
        },
        utils::{config::Config, downloader::ModelPaths, progress::ProgressReporter},
    };

    #[test]
    fn raw_nvfp4_config_constructs_and_prefills() -> anyhow::Result<()> {
        let Some(path) = std::env::var_os("XINFER_QWEN3_NVFP4_CHECKPOINT") else {
            eprintln!("set XINFER_QWEN3_NVFP4_CHECKPOINT to run the real-model smoke test");
            return Ok(());
        };
        let directory = Path::new(&path);
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let raw_quant = config.quantization_config.as_ref().unwrap();
        anyhow::ensure!(raw_quant.quant_method == "compressed-tensors");
        anyhow::ensure!(raw_quant.group_size == 0 && raw_quant.bits == 0);

        let device = xinfer::utils::new_device(0)?;
        let paths = ModelPaths {
            tokenizer_filename: directory.join("tokenizer.json"),
            tokenizer_config_filename: directory.join("tokenizer_config.json"),
            config_filename: directory.join("config.json"),
            generation_config_filename: directory.join("generation_config.json"),
            filenames: vec![directory.join("model.safetensors")],
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
        // No explicit call to normalize_quantization_config: the public model
        // constructor must do this before loading any quantized projection.
        let model = Qwen3ForCausalLM::new(
            &vb,
            Rc::new(Comm::default()),
            &config,
            DType::BF16,
            false,
            &device,
            Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        )?;
        let shape = (1, 16, config.num_key_value_heads, config.head_dim.unwrap());
        let cache = (0..config.num_hidden_layers)
            .map(|_| {
                Ok((
                    Tensor::zeros(shape, DType::BF16, &device)?,
                    Tensor::zeros(shape, DType::BF16, &device)?,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let input_ids = [151643u32, 3837, 264];
        let positions = [0i64, 1, 2];
        let metadata = InputMetadata {
            is_prefill: true,
            is_mla: false,
            sequence_ids: Some(vec![0]),
            mamba_slot_mapping: None,
            slot_mapping: Tensor::from_slice(&positions, (3,), &device)?,
            block_tables: Some(Tensor::from_vec(vec![0u32], (1, 1), &device)?),
            block_tables_host: Some(vec![vec![0]]),
            context_lens_host: Some(vec![3]),
            context_lens: Some(Tensor::from_vec(vec![3u32], (1,), &device)?),
            cu_seqlens_q: Some(Tensor::from_vec(vec![0u32, 3], (2,), &device)?),
            cu_seqlens_k: Some(Tensor::from_vec(vec![0u32, 3], (2,), &device)?),
            max_seqlen_q: 3,
            max_seqlen_k: 3,
            max_context_len: 3,
            seqlens: Some(vec![3]),
            flashinfer_metadata: None,
            is_mtp_verify: false,
        };
        let _guard = xinfer::models::layers::linear::set_linear_is_prefill(true);
        let logits = model.forward(
            &Tensor::from_slice(&input_ids, (3,), &device)?,
            &Tensor::from_slice(&positions, (3,), &device)?,
            Some(&cache),
            &metadata,
            false,
        )?;
        let values = logits.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(values.len() == config.vocab_size.unwrap());
        anyhow::ensure!(values.iter().all(|value| value.is_finite()));
        Ok(())
    }
}
