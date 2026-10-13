//! Opt-in GB10-scale restore and continuation check. The checkpoint is read
//! from XINFER_PACKED_KV_BATCHING_QWEN3_4B; no fixture files are modified.

#[cfg(feature = "cuda")]
mod cuda_test {
    use anyhow::{ensure, Result};
    use attention_rs::InputMetadata;
    use candle_core::{DType, Device, Tensor};
    use parking_lot::RwLock;
    use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
    use xinfer::{
        models::{
            layers::{distributed::Comm, VarBuilderX},
            qwen3::Qwen3ForCausalLM,
        },
        utils::{
            config::Config,
            downloader::ModelPaths,
            packed_kv_restore::{
                encode_dense_packed_kv_tile_sealed,
                restore_dense_packed_kv_pages_sealed_with_options, DensePackedKvRestoreOptions,
                PackedKvAxis, SealedDensePackedKvPage, SealedDensePackedKvRestore,
            },
            progress::ProgressReporter,
            GpuKvCache,
        },
    };

    const PREFIX: usize = 8192;
    const BLOCK: usize = 16;
    const CONTINUATION: usize = 16;

    fn cache(config: &Config, device: &Device) -> Result<GpuKvCache> {
        let shape = (
            PREFIX / BLOCK + 1,
            BLOCK,
            config.num_key_value_heads,
            config.head_dim.unwrap(),
        );
        Ok(GpuKvCache::Flash(
            (0..config.num_hidden_layers)
                .map(|_| {
                    Ok((
                        Tensor::zeros(shape, DType::BF16, device)?,
                        Tensor::zeros(shape, DType::BF16, device)?,
                    ))
                })
                .collect::<candle_core::Result<Vec<_>>>()?,
        ))
    }

    fn sample_page(
        axis: PackedKvAxis,
        bits: u8,
        exact: bool,
        heads: usize,
        channels: usize,
    ) -> Result<SealedDensePackedKvPage> {
        let words = (0..BLOCK * heads * channels)
            .map(|index| half::bf16::from_f32(((index * 13 % 97) as f32 - 48.0) / 32.0).to_bits())
            .collect::<Vec<_>>();
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(encode_dense_packed_kv_tile_sealed(
            &bytes,
            BLOCK,
            heads,
            channels,
            axis,
            bits,
            if exact { BLOCK } else { 0 },
        )?
        .pop()
        .unwrap())
    }

    fn continuation(
        model: &Qwen3ForCausalLM,
        cache: &GpuKvCache,
        device: &Device,
    ) -> Result<Vec<Vec<f32>>> {
        let _guard = xinfer::models::layers::linear::set_linear_is_prefill(false);
        let block_table = (0..=PREFIX / BLOCK)
            .map(|block| block as u32)
            .collect::<Vec<_>>();
        let mut logits = Vec::new();
        for step in 0..CONTINUATION {
            let position = PREFIX + step;
            let length = (position + 1) as u32;
            let metadata = InputMetadata {
                is_prefill: false,
                is_mla: false,
                sequence_ids: Some(vec![11]),
                mamba_slot_mapping: None,
                slot_mapping: Tensor::from_vec(vec![position as i64], (1,), device)?,
                block_tables: Some(Tensor::from_vec(
                    block_table.clone(),
                    (1, block_table.len()),
                    device,
                )?),
                block_tables_host: Some(vec![block_table.clone()]),
                context_lens_host: Some(vec![length]),
                context_lens: Some(Tensor::from_vec(vec![length], (1,), device)?),
                cu_seqlens_q: None,
                cu_seqlens_k: None,
                max_seqlen_q: 0,
                max_seqlen_k: 0,
                max_context_len: position + 1,
                seqlens: None,
                flashinfer_metadata: None,
                is_mtp_verify: false,
            };
            logits.push(
                model
                    .forward(
                        &Tensor::from_vec(vec![3837u32], (1,), device)?,
                        &Tensor::from_vec(vec![position as i64], (1,), device)?,
                        cache.as_pairs(),
                        &metadata,
                        false,
                    )?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        Ok(logits)
    }

    #[test]
    #[ignore = "GB10 8192-token Qwen3-4B checkpoint restore benchmark"]
    fn qwen3_4b_8192_mixed_restore_batching_and_logits() -> Result<()> {
        let directory = std::env::var("XINFER_PACKED_KV_BATCHING_QWEN3_4B")?;
        let directory = Path::new(&directory);
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        ensure!(config.num_hidden_layers == 36 && config.num_key_value_heads == 8);
        let device = Device::new_cuda(0)?;
        let mut filenames = std::fs::read_dir(directory)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("model-") && name.ends_with(".safetensors")
                    })
            })
            .collect::<Vec<_>>();
        filenames.sort();
        ensure!(!filenames.is_empty(), "missing checkpoint shards");
        let paths = ModelPaths {
            tokenizer_filename: directory.join("tokenizer.json"),
            tokenizer_config_filename: directory.join("tokenizer_config.json"),
            config_filename: directory.join("config.json"),
            generation_config_filename: directory.join("generation_config.json"),
            filenames,
            auxiliary_filenames: vec![],
            chat_template_filename: None,
        };
        let vb = VarBuilderX::new(&paths, false, DType::BF16, &device)?;
        let model = Qwen3ForCausalLM::new(
            &vb,
            Rc::new(Comm::default()),
            &config,
            DType::BF16,
            false,
            &device,
            Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        )?;
        let pages = [
            sample_page(PackedKvAxis::Key, 2, false, 8, 128)?,
            sample_page(PackedKvAxis::Value, 4, false, 8, 128)?,
            sample_page(PackedKvAxis::Key, 2, true, 8, 128)?,
            sample_page(PackedKvAxis::Value, 4, true, 8, 128)?,
        ];
        let oracles = pages
            .iter()
            .map(|page| page.page().decode_cpu_bf16())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let requests = (0..config.num_hidden_layers)
            .flat_map(|layer| {
                (0..2).flat_map(move |axis| {
                    (0..PREFIX / BLOCK).map(move |block| (layer, axis, block))
                })
            })
            .map(|(layer, axis, block)| SealedDensePackedKvRestore {
                layer,
                block,
                token_offset: 0,
                page: &pages[axis + usize::from(block >= PREFIX / BLOCK - 2) * 2],
            })
            .collect::<Vec<_>>();
        let mut reference_logits: Option<Vec<Vec<f32>>> = None;
        for page_limit in [128, 512, 1024] {
            let options = DensePackedKvRestoreOptions {
                max_batch_pages: page_limit,
                max_batch_payload_bytes: 64 * 1024 * 1024,
            };
            let mut samples = Vec::new();
            for repeat in 0..3 {
                let gpu_cache = cache(&config, &device)?;
                let started = Instant::now();
                let stats = restore_dense_packed_kv_pages_sealed_with_options(
                    &gpu_cache, &requests, options,
                )?;
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                ensure!(stats.pages == 36_864);
                ensure!(stats.kernel_launches == 36_864 / page_limit);
                ensure!(stats.staging_uploads == stats.kernel_launches);
                ensure!(stats.batch_synchronizations == stats.kernel_launches);
                ensure!(stats.staging_host_allocations == 1);
                ensure!(stats.staging_device_allocations == 1);
                ensure!(stats.max_batch_pages == page_limit);
                ensure!(stats.max_batch_payload_bytes <= options.max_batch_payload_bytes);
                ensure!(
                    stats.max_batch_staging_bytes
                        <= options.max_batch_payload_bytes + page_limit * 64 + 7
                );
                if repeat == 0 {
                    for (layer, (key, value)) in gpu_cache.as_pairs().unwrap().iter().enumerate() {
                        for (axis, tensor) in [key, value].into_iter().enumerate() {
                            let actual = tensor
                                .to_device(&Device::Cpu)?
                                .flatten_all()?
                                .to_vec1::<half::bf16>()?;
                            let page_words = BLOCK * 8 * 128;
                            for block in 0..PREFIX / BLOCK {
                                let source = axis + usize::from(block >= PREFIX / BLOCK - 2) * 2;
                                let start = block * page_words;
                                ensure!(
                                    actual[start..start + page_words]
                                        .iter()
                                        .map(|word| word.to_bits())
                                        .eq(oracles[source].iter().copied()),
                                    "BF16 parity failed at layer {layer} axis {axis} block {block}"
                                );
                            }
                        }
                    }
                    let logits = continuation(&model, &gpu_cache, &device)?;
                    if let Some(reference) = &reference_logits {
                        ensure!(
                            &logits == reference,
                            "batch limit changed continuation logits"
                        );
                    } else {
                        reference_logits = Some(logits);
                    }
                }
                samples.push(elapsed);
                eprintln!("qwen3_4b_8192 page_limit={page_limit} repeat={repeat} restore_ms={elapsed:.3} launches={} uploads={} host_allocs={} device_allocs={} syncs={} max_pages={} max_payload={} max_staging={} packed_bytes={}", stats.kernel_launches, stats.staging_uploads, stats.staging_host_allocations, stats.staging_device_allocations, stats.batch_synchronizations, stats.max_batch_pages, stats.max_batch_payload_bytes, stats.max_batch_staging_bytes, stats.packed_bytes);
            }
            eprintln!(
                "qwen3_4b_8192 page_limit={page_limit} first_ms={:.3} later_ms={:.3}",
                samples[0],
                (samples[1] + samples[2]) / 2.0
            );
        }
        Ok(())
    }
}
