//! Real-checkpoint exact control for the opt-in Qwen3 attention probe.
//! XINFER_QWEN3_ATTENTION_PROBE_CHECKPOINT points to a BF16 Qwen3 directory.

#[cfg(feature = "cuda")]
mod cuda_test {
    use anyhow::{ensure, Result};
    use attention_rs::InputMetadata;
    use candle_core::{DType, Device, Tensor};
    use parking_lot::{Mutex, RwLock};
    use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
    use xinfer::{
        models::{
            layers::{
                attention::{AttentionProbe, AttentionProbeObserver},
                distributed::Comm,
                sparse_attention::{SparseAttentionHead, SparseAttentionProvider},
                VarBuilderX,
            },
            qwen3::Qwen3ForCausalLM,
        },
        utils::{
            config::Config, downloader::ModelPaths, kv_prefix_export::export_bf16_kv_prefix,
            progress::ProgressReporter,
        },
    };

    const BLOCK: usize = 16;

    fn load_model(directory: &Path) -> Result<(Device, Config, Qwen3ForCausalLM)> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        ensure!(config.attn_logit_softcapping.is_none());
        ensure!(config.sliding_window.is_none());
        let device = Device::new_cuda(0)?;
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
        let model = Qwen3ForCausalLM::new(
            &vb,
            Rc::new(Comm::default()),
            &config,
            DType::BF16,
            false,
            &device,
            Arc::new(RwLock::new(Box::new(ProgressReporter::new(0)))),
        )?;
        Ok((device, config, model))
    }

    fn cache(config: &Config, device: &Device) -> Result<Vec<(Tensor, Tensor)>> {
        let shape = (
            2,
            BLOCK,
            config.num_key_value_heads,
            config.head_dim.unwrap(),
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

    fn forward(
        model: &Qwen3ForCausalLM,
        device: &Device,
        cache: &Vec<(Tensor, Tensor)>,
        ids: &[u32],
        positions: &[i64],
        slots: &[i64],
        lengths: [u32; 2],
        is_prefill: bool,
    ) -> Result<Vec<f32>> {
        let _guard = xinfer::models::layers::linear::set_linear_is_prefill(is_prefill);
        let count = ids.len();
        let metadata = InputMetadata {
            is_prefill,
            is_mla: false,
            sequence_ids: Some(vec![11, 22]),
            mamba_slot_mapping: None,
            slot_mapping: Tensor::from_slice(slots, (count,), device)?,
            block_tables: Some(Tensor::from_vec(vec![0u32, 1], (2, 1), device)?),
            block_tables_host: Some(vec![vec![0], vec![1]]),
            context_lens_host: Some(lengths.to_vec()),
            context_lens: Some(Tensor::from_slice(&lengths, (2,), device)?),
            cu_seqlens_q: is_prefill
                .then(|| Tensor::from_vec(vec![0u32, 3, 5], (3,), device))
                .transpose()?,
            cu_seqlens_k: is_prefill
                .then(|| Tensor::from_vec(vec![0u32, 3, 5], (3,), device))
                .transpose()?,
            max_seqlen_q: if is_prefill { 3 } else { 0 },
            max_seqlen_k: if is_prefill { 3 } else { 0 },
            max_context_len: lengths.into_iter().max().unwrap() as usize,
            seqlens: is_prefill.then_some(vec![3, 5]),
            flashinfer_metadata: None,
            is_mtp_verify: false,
        };
        Ok(model
            .forward(
                &Tensor::from_slice(ids, (count,), device)?,
                &Tensor::from_slice(positions, (count,), device)?,
                Some(cache),
                &metadata,
                false,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    fn cpu_control(
        probe: &AttentionProbe,
        cache: &(Tensor, Tensor),
        row_sequences: &[usize],
    ) -> Result<f32> {
        let positions = probe.positions.flatten_all()?.to_vec1::<i64>()?;
        ensure!(positions.len() == row_sequences.len());
        let q = probe
            .query
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = probe
            .context
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Use the public portable exporter, including its BF16 wire bytes,
        // instead of reading Candle's cache storage directly.
        let exported = export_bf16_kv_prefix(&[cache.clone()], 2 * BLOCK)?;
        let decode = |bytes: &[u8]| {
            bytes
                .chunks_exact(2)
                .map(|word| half::bf16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32())
                .collect::<Vec<_>>()
        };
        let keys = decode(&exported[0]);
        let values = decode(&exported[1]);
        let heads = probe.query_heads;
        let kv_heads = probe.kv_heads;
        let width = probe.head_width;
        ensure!(heads % kv_heads == 0);
        ensure!(q.len() == positions.len() * heads * width);
        ensure!(actual.len() == q.len());
        let mut max_abs = 0f32;
        for (row, &sequence) in row_sequences.iter().enumerate() {
            let position = usize::try_from(positions[row])?;
            ensure!(position < BLOCK);
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                let query = &q[(row * heads + head) * width..(row * heads + head + 1) * width];
                let scores = (0..=position)
                    .map(|token| {
                        let base = ((sequence * BLOCK + token) * kv_heads + kv_head) * width;
                        query
                            .iter()
                            .zip(&keys[base..base + width])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            * probe.score_scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let masses = scores
                    .iter()
                    .map(|s| (s - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = masses.iter().sum::<f32>();
                for channel in 0..width {
                    let expected = masses
                        .iter()
                        .enumerate()
                        .map(|(token, mass)| {
                            let base = ((sequence * BLOCK + token) * kv_heads + kv_head) * width;
                            mass * values[base + channel]
                        })
                        .sum::<f32>()
                        / denominator;
                    let index = (row * heads + head) * width + channel;
                    max_abs = max_abs.max((expected - actual[index]).abs());
                }
            }
        }
        Ok(max_abs)
    }

    fn sparse_provider(
        cache: &(Tensor, Tensor),
        selected_old: Vec<usize>,
    ) -> SparseAttentionProvider {
        let cache = cache.clone();
        Arc::new(move |probe| {
            if probe.layer_index != 0 || probe.is_prefill {
                return Ok(Vec::new());
            }
            let query_position =
                usize::try_from(probe.positions.flatten_all()?.to_vec1::<i64>()?[0])
                    .map_err(candle_core::Error::wrap)?;
            let positions = selected_old
                .iter()
                .copied()
                .chain(2..=query_position)
                .collect::<Vec<_>>();
            let exported = export_bf16_kv_prefix(&[cache.clone()], 2 * BLOCK)?;
            let width = probe.head_width;
            let kv_heads = probe.kv_heads;
            let select = |bytes: &[u8]| {
                positions
                    .iter()
                    .flat_map(|&position| {
                        (0..width).map(move |channel| {
                            let word = (position * kv_heads * width + channel) * 2;
                            half::bf16::from_bits(u16::from_le_bytes([
                                bytes[word],
                                bytes[word + 1],
                            ]))
                        })
                    })
                    .collect::<Vec<_>>()
            };
            let count = positions.len();
            let keys = select(&exported[0]);
            let values = select(&exported[1]);
            Ok(vec![SparseAttentionHead {
                row: 0,
                query_head: 0,
                kv_head: 0,
                prefix_end: 3,
                exact_start: 2,
                positions,
                keys: Tensor::from_vec(keys, (count, width), &Device::Cpu)?,
                values: Tensor::from_vec(values, (count, width), &Device::Cpu)?,
            }])
        })
    }

    fn cpu_selected_head(
        probe: &AttentionProbe,
        cache: &(Tensor, Tensor),
        positions: &[usize],
    ) -> Result<Vec<f32>> {
        let exported = export_bf16_kv_prefix(&[cache.clone()], 2 * BLOCK)?;
        let width = probe.head_width;
        let query = probe
            .query
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let query = &query[..width];
        let value = |tile: usize, position: usize, channel: usize| {
            let offset = (position * probe.kv_heads * width + channel) * 2;
            let bytes = &exported[tile];
            half::bf16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
        };
        let scores = positions
            .iter()
            .map(|&position| {
                (0..width)
                    .map(|channel| {
                        f64::from(query[channel]) * f64::from(value(0, position, channel))
                    })
                    .sum::<f64>()
                    * f64::from(probe.score_scale)
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = mass.iter().sum::<f64>();
        Ok((0..width)
            .map(|channel| {
                (positions
                    .iter()
                    .enumerate()
                    .map(|(index, &position)| mass[index] * f64::from(value(1, position, channel)))
                    .sum::<f64>()
                    / denominator) as f32
            })
            .collect())
    }

    fn max_abs(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    }

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }

    fn prefill_and_decode(
        model: &Qwen3ForCausalLM,
        device: &Device,
        cache: &Vec<(Tensor, Tensor)>,
    ) -> Result<Vec<f32>> {
        let _ = forward(
            model,
            device,
            cache,
            &[151643, 3837, 264, 151643, 264],
            &[0, 1, 2, 0, 1],
            &[0, 1, 2, 16, 17],
            [3, 2],
            true,
        )?;
        forward(
            model,
            device,
            cache,
            &[3837, 3837],
            &[3, 2],
            &[3, 18],
            [4, 3],
            false,
        )
    }

    #[test]
    fn real_qwen3_sparse_head_all_selected_and_subset_controls() -> Result<()> {
        let Some(path) = std::env::var_os("XINFER_QWEN3_ATTENTION_PROBE_CHECKPOINT") else {
            return Ok(());
        };
        let (device, config, model) = load_model(Path::new(&path))?;
        let baseline_cache = cache(&config, &device)?;
        let baseline = prefill_and_decode(&model, &device, &baseline_cache)?;

        let all_cache = cache(&config, &device)?;
        let all_probes = Arc::new(Mutex::new(Vec::<AttentionProbe>::new()));
        let collected = all_probes.clone();
        model.set_attention_probe_observer(Some(Arc::new(move |probe| {
            if probe.layer_index == 0 && !probe.is_prefill {
                collected.lock().push(probe);
            }
            Ok(())
        })));
        model.set_sparse_attention_provider(Some(sparse_provider(&all_cache[0], vec![0, 1])));
        let all_logits = prefill_and_decode(&model, &device, &all_cache)?;
        model.set_sparse_attention_provider(None);
        model.set_attention_probe_observer(None);
        let all_probe = all_probes.lock().pop().unwrap();
        let native = all_probe
            .context
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let effective = all_probe
            .effective_context
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let width = all_probe.head_width;
        let identity_context_error = max_abs(&native[..width], &effective[..width]);
        ensure!(identity_context_error < 0.01);
        ensure!(native[width..] == effective[width..]);
        let identity_logit_error = max_abs(&baseline, &all_logits);
        ensure!(identity_logit_error < 0.01);
        let vocab = config.vocab_size.unwrap();
        for row in 0..2 {
            ensure!(
                argmax(&baseline[row * vocab..(row + 1) * vocab])
                    == argmax(&all_logits[row * vocab..(row + 1) * vocab])
            );
        }

        let subset_cache = cache(&config, &device)?;
        let subset_probes = Arc::new(Mutex::new(Vec::<AttentionProbe>::new()));
        let collected = subset_probes.clone();
        model.set_attention_probe_observer(Some(Arc::new(move |probe| {
            if probe.layer_index == 0 && !probe.is_prefill {
                collected.lock().push(probe);
            }
            Ok(())
        })));
        model.set_sparse_attention_provider(Some(sparse_provider(&subset_cache[0], vec![0])));
        let subset_logits = prefill_and_decode(&model, &device, &subset_cache)?;
        model.set_sparse_attention_provider(None);
        model.set_attention_probe_observer(None);
        let subset_probe = subset_probes.lock().pop().unwrap();
        let expected = cpu_selected_head(&subset_probe, &subset_cache[0], &[0, 2, 3])?;
        let actual = subset_probe
            .effective_context
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let subset_context_error = max_abs(&expected, &actual[..width]);
        ensure!(subset_context_error < 0.01);
        let untouched = subset_probe
            .context
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(untouched[width..] == actual[width..]);
        let subset_logit_change = max_abs(&baseline, &subset_logits);
        ensure!(subset_logit_change > 0.0);
        eprintln!(
            "qwen3_sparse_control identity_context_max_abs={identity_context_error:.6} identity_logit_max_abs={identity_logit_error:.6} subset_oracle_max_abs={subset_context_error:.6} subset_logit_change={subset_logit_change:.6}"
        );
        Ok(())
    }

    #[test]
    fn real_qwen3_query_and_context_match_exported_rotated_cache() -> Result<()> {
        let Some(path) = std::env::var_os("XINFER_QWEN3_ATTENTION_PROBE_CHECKPOINT") else {
            return Ok(());
        };
        let directory = Path::new(&path);
        let (device, config, model) = load_model(directory)?;
        // Compile and warm the CUDA kernels before the paired timings.
        let warmup_cache = cache(&config, &device)?;
        let _ = forward(
            &model,
            &device,
            &warmup_cache,
            &[151643, 3837, 264, 151643, 264],
            &[0, 1, 2, 0, 1],
            &[0, 1, 2, 16, 17],
            [3, 2],
            true,
        )?;
        let _ = forward(
            &model,
            &device,
            &warmup_cache,
            &[3837, 3837],
            &[3, 2],
            &[3, 18],
            [4, 3],
            false,
        )?;
        let probes = Arc::new(Mutex::new(Vec::<AttentionProbe>::new()));
        let collected = probes.clone();
        let observer: AttentionProbeObserver = Arc::new(move |probe| {
            if probe.layer_index == 0 {
                collected.lock().push(probe);
            }
            Ok(())
        });
        model.set_attention_probe_observer(Some(observer));
        let observed_cache = cache(&config, &device)?;
        let began = Instant::now();
        let prefill_logits = forward(
            &model,
            &device,
            &observed_cache,
            &[151643, 3837, 264, 151643, 264],
            &[0, 1, 2, 0, 1],
            &[0, 1, 2, 16, 17],
            [3, 2],
            true,
        )?;
        let decode_logits = forward(
            &model,
            &device,
            &observed_cache,
            &[3837, 3837],
            &[3, 2],
            &[3, 18],
            [4, 3],
            false,
        )?;
        let observed_ms = began.elapsed().as_secs_f64() * 1000.0;
        let captured = probes.lock();
        ensure!(captured.len() == 2);
        ensure!(captured[0].layer_index == 0 && captured[0].is_prefill);
        ensure!(captured[1].layer_index == 0 && !captured[1].is_prefill);
        ensure!(captured[0].sequence_ids.as_deref() == Some(&[11, 22][..]));
        ensure!(captured[0].seqlens.as_deref() == Some(&[3, 5][..]));
        ensure!(captured[0].query_heads == config.num_attention_heads);
        ensure!(captured[0].kv_heads == config.num_key_value_heads);
        ensure!(captured[0].head_width == config.head_dim.unwrap());
        ensure!(captured[0].dtype == DType::BF16);
        let prefill_error = cpu_control(&captured[0], &observed_cache[0], &[0, 0, 0, 1, 1])?;
        let decode_error = cpu_control(&captured[1], &observed_cache[0], &[0, 1])?;
        drop(captured);
        ensure!(
            prefill_error < 0.01,
            "prefill CPU control max abs {prefill_error}"
        );
        ensure!(
            decode_error < 0.01,
            "decode CPU control max abs {decode_error}"
        );

        model.set_attention_probe_observer(None);
        let plain_cache = cache(&config, &device)?;
        let began = Instant::now();
        let plain_prefill = forward(
            &model,
            &device,
            &plain_cache,
            &[151643, 3837, 264, 151643, 264],
            &[0, 1, 2, 0, 1],
            &[0, 1, 2, 16, 17],
            [3, 2],
            true,
        )?;
        let plain_decode = forward(
            &model,
            &device,
            &plain_cache,
            &[3837, 3837],
            &[3, 2],
            &[3, 18],
            [4, 3],
            false,
        )?;
        let plain_ms = began.elapsed().as_secs_f64() * 1000.0;
        ensure!(prefill_logits == plain_prefill);
        ensure!(decode_logits == plain_decode);
        ensure!(probes.lock().len() == 2);
        eprintln!(
            "qwen3_attention_probe prefill_max_abs={prefill_error:.6} decode_max_abs={decode_error:.6} observed_ms={observed_ms:.3} plain_ms={plain_ms:.3}"
        );
        Ok(())
    }
}
