use super::*;
use crate::models::layers::distributed::ReplicatedLinear;
use candle_core::Device;

#[cfg(feature = "nemotron-mamba-ssd-wmma")]
#[test]
fn ssd_wmma_matches_recurrence_with_nonzero_starts_and_partial_chunk() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?) != Some(121) {
        return Ok(());
    }
    const TOKENS: usize = 257;
    const HEADS: usize = 64;
    const GROUPS: usize = 8;
    const HD: usize = 64;
    const SD: usize = 128;
    const CT: usize = 128;
    const CONV: usize = HEADS * HD + 2 * GROUPS * SD;
    let chunks = TOKENS.div_ceil(CT);
    let mut conv = vec![0f32; TOKENS * CONV];
    for t in 0..TOKENS {
        for i in 0..CONV {
            conv[t * CONV + i] = ((t * 17 + i * 11) % 37) as f32 * 0.002 - 0.035;
        }
    }
    let mut dt_decay = vec![0f32; TOKENS * HEADS * 2];
    for t in 0..TOKENS {
        for h in 0..HEADS {
            dt_decay[(t * HEADS + h) * 2] = 0.18 + ((t + h) % 9) as f32 * 0.01;
            dt_decay[(t * HEADS + h) * 2 + 1] = if t % CT == 49 && h % 3 == 0 {
                0.0
            } else {
                0.91 + (h % 7) as f32 * 0.003
            };
        }
    }
    let starts = (0..chunks * HEADS * HD * SD)
        .map(|i| ((i * 13) % 19) as f32 * 0.0003 - 0.002)
        .collect::<Vec<_>>();
    let d = (0..HEADS)
        .map(|h| 0.1 + h as f32 * 0.001)
        .collect::<Vec<_>>();
    let mut expected = vec![0f32; TOKENS * HEADS * HD];
    for chunk in 0..chunks {
        for head in 0..HEADS {
            let group = head / (HEADS / GROUPS);
            for dim in 0..HD {
                let start_offset = ((chunk * HEADS + head) * HD + dim) * SD;
                let mut state = starts[start_offset..start_offset + SD].to_vec();
                for t in chunk * CT..((chunk + 1) * CT).min(TOKENS) {
                    let dt = dt_decay[(t * HEADS + head) * 2];
                    let decay = dt_decay[(t * HEADS + head) * 2 + 1];
                    let x = conv[t * CONV + head * HD + dim];
                    let mut sum = 0f32;
                    for s in 0..SD {
                        let b = conv[t * CONV + HEADS * HD + group * SD + s];
                        let c = conv[t * CONV + HEADS * HD + GROUPS * SD + group * SD + s];
                        state[s] = decay * state[s] + dt * x * b;
                        sum += state[s] * c;
                    }
                    expected[t * HEADS * HD + head * HD + dim] = sum + x * d[head];
                }
            }
        }
    }
    let conv = Tensor::from_vec(conv, (TOKENS, CONV), &device)?;
    let dt_decay = Tensor::from_vec(dt_decay, (TOKENS, HEADS, 2), &device)?;
    let starts = Tensor::from_vec(starts, (chunks, HEADS, HD, SD), &device)?;
    let d = Tensor::from_vec(d, HEADS, &device)?;
    let cb = unsafe { Tensor::empty_((chunks, GROUPS, CT, CT), DType::BF16, &device)? };
    let log_decay = unsafe { Tensor::empty_((TOKENS, HEADS), DType::F32, &device)? };
    let reset_at = unsafe { Tensor::empty_((TOKENS, HEADS), DType::F32, &device)? };
    let transposed = unsafe { Tensor::empty_((chunks, HEADS, SD, HD), DType::BF16, &device)? };
    let output = unsafe { Tensor::empty_((TOKENS, HEADS * HD), DType::F32, &device)? };
    let cuda = device.as_cuda_device()?.cuda_device();
    let launch = |cb: &Tensor,
                  cb64: i32,
                  k64: i32,
                  direct_starts: i32,
                  output_rows: i32,
                  output_k128: i32| unsafe {
        xinfer_nemotron_ssd_wmma(
            pointer(&conv).unwrap() as *const f32,
            pointer(&dt_decay).unwrap() as *const f32,
            pointer(&starts).unwrap() as *const f32,
            pointer(&d).unwrap() as *const f32,
            pointer(&output).unwrap() as *mut f32,
            pointer(&cb).unwrap() as *mut std::ffi::c_void,
            pointer(&log_decay).unwrap() as *mut f32,
            pointer(&reset_at).unwrap() as *mut f32,
            if direct_starts != 0 {
                std::ptr::null_mut()
            } else {
                pointer(&transposed).unwrap() as *mut std::ffi::c_void
            },
            TOKENS as i32,
            HEADS as i32,
            GROUPS as i32,
            CONV as i32,
            cb64,
            k64,
            direct_starts,
            output_rows,
            output_k128,
            *cuda.cu_stream() as i64,
        )
    };
    assert_eq!(launch(&cb, 0, 0, 0, 64, 0), 0);
    let actual = output.flatten_all()?.to_vec1::<f32>()?;
    let cb_bits = cb
        .flatten_all()?
        .to_vec1::<half::bf16>()?
        .into_iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs < 0.003, "SSD WMMA max abs {max_abs}");
    // Poison the skipped upper-right tile in a fresh allocation. The output
    // must be independent of those never-written future-token products.
    let cb_tri = Tensor::from_vec(
        vec![half::bf16::from_f32(f32::NAN); chunks * GROUPS * CT * CT],
        (chunks, GROUPS, CT, CT),
        &device,
    )?;
    assert_eq!(launch(&cb_tri, 1, 0, 0, 64, 0), 0);
    let tri_bits = cb_tri
        .flatten_all()?
        .to_vec1::<half::bf16>()?
        .into_iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    for (index, (&reference, &triangular)) in cb_bits.iter().zip(&tri_bits).enumerate() {
        let row = index / CT % CT;
        let col = index % CT;
        if col <= row {
            assert_eq!(reference, triangular, "C×B row {row}, col {col}");
        } else if row < 64 && col >= 64 {
            assert!(half::bf16::from_bits(triangular).is_nan());
        }
    }
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 0, 0, 64, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 0, 64, 0), 0);
    let tri_bits = cb_tri
        .flatten_all()?
        .to_vec1::<half::bf16>()?
        .into_iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    for (index, (&reference, &triangular)) in cb_bits.iter().zip(&tri_bits).enumerate() {
        if index % CT <= index / CT % CT {
            assert_eq!(reference, triangular);
        }
    }
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 0, 64, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 1, 64, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 1, 64, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 0, 1, 64, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb, 0, 0, 0, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 0, 0, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 0, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 0, 1, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 1, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 1, 128, 0), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 1, 128, 1), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_eq!(launch(&cb_tri, 1, 1, 0, 128, 1), 0);
    assert_eq!(actual, output.flatten_all()?.to_vec1::<f32>()?);
    assert_ne!(launch(&cb_tri, 1, 0, 1, 128, 1), 0);
    Ok(())
}

fn fixture(device: &Device, head_dim: usize) -> Result<(Mamba2, Tensor)> {
    fixture_with_tokens(device, head_dim, 6)
}

fn fixture_with_tokens(
    device: &Device,
    head_dim: usize,
    tokens: usize,
) -> Result<(Mamba2, Tensor)> {
    let heads = 8;
    let groups = 2;
    let state_dim = 128;
    let kernel = 4;
    let inner = heads * head_dim;
    let conv_dim = inner + 2 * groups * state_dim;
    let in_dim = inner + conv_dim + heads;
    let in_proj = ReplicatedLinear::from_weight_bias(
        Tensor::zeros((in_dim, inner), DType::F32, device)?,
        None,
    )?;
    let mut identity = vec![0f32; inner * inner];
    for i in 0..inner {
        identity[i * inner + i] = 1.0;
    }
    let out_proj = ReplicatedLinear::from_weight_bias(
        Tensor::from_vec(identity, (inner, inner), device)?,
        None,
    )?;
    let conv_weight = Tensor::from_vec(
        (0..kernel * conv_dim)
            .map(|index| (index / conv_dim + 1) as f32 * 0.1)
            .collect::<Vec<_>>(),
        (kernel, conv_dim),
        device,
    )?;
    let mixer = Mamba2 {
        in_proj,
        out_proj,
        conv_weight,
        conv_bias: Tensor::from_vec(vec![0.03f32; conv_dim], (conv_dim,), device)?,
        dt_bias: Tensor::from_vec(vec![0.1f32; heads], (heads,), device)?,
        a: Tensor::from_vec(vec![-0.5f32; heads], (heads,), device)?,
        d: Tensor::from_vec(vec![0.2f32; heads], (heads,), device)?,
        norm_weight: Tensor::ones((inner,), DType::F32, device)?,
        heads,
        head_dim,
        groups,
        state_dim,
        kernel,
        conv_dim,
        eps: 1e-5,
        dt_limit: (0.0, 2.0),
        device: device.clone(),
        dtype: DType::F32,
    };
    let mut projected = vec![0f32; tokens * in_dim];
    for t in 0..tokens {
        for i in 0..inner {
            projected[t * in_dim + i] = 0.2 + (i % 13) as f32 * 0.02;
        }
        for i in 0..conv_dim {
            projected[t * in_dim + inner + i] = 0.1 + ((t * 17 + i * 7) % 29) as f32 * 0.007;
        }
        for h in 0..heads {
            projected[t * in_dim + inner + conv_dim + h] = -0.3 + ((t + h) % 7) as f32 * 0.1;
        }
    }
    Ok((
        mixer,
        Tensor::from_vec(projected, (tokens, in_dim), device)?,
    ))
}

#[test]
fn span_scan_matches_steps_and_keeps_continued_state() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };
    let (mixer, projected) = fixture(&device, 8)?;
    let mut initial = mixer.zero_state()?;
    initial.conv = Tensor::from_vec(
        (0..(mixer.kernel - 1) * mixer.conv_dim)
            .map(|i| (i % 23) as f32 * 0.003)
            .collect::<Vec<_>>(),
        (mixer.kernel - 1, mixer.conv_dim),
        &device,
    )?;
    initial.ssm = Tensor::from_vec(
        (0..mixer.heads * mixer.head_dim * mixer.state_dim)
            .map(|i| (i % 31) as f32 * 0.001)
            .collect::<Vec<_>>(),
        (mixer.heads, mixer.head_dim, mixer.state_dim),
        &device,
    )?;
    let mut step_state = initial.clone();
    let mut step_outputs = Vec::new();
    for t in 0..6 {
        step_outputs.push(mixer.step(&projected.get(t)?, &mut step_state)?);
    }
    let step_refs = step_outputs.iter().collect::<Vec<_>>();
    let expected = Tensor::cat(&step_refs, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut fast_state = initial.clone();
    let actual = run(&mixer, &projected, &mut fast_state)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_output = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let expected_state = step_state.ssm.flatten_all()?.to_vec1::<f32>()?;
    let actual_state = fast_state.ssm.flatten_all()?.to_vec1::<f32>()?;
    let max_state = expected_state
        .iter()
        .zip(&actual_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_output < 5e-3, "output max abs {max_output}");
    assert!(max_state < 5e-5, "state max abs {max_state}");
    assert_eq!(
        step_state.conv.flatten_all()?.to_vec1::<f32>()?,
        fast_state.conv.flatten_all()?.to_vec1::<f32>()?
    );

    let mut continued = initial.clone();
    run(&mixer, &projected.narrow(0, 0, 2)?, &mut continued)?;
    run(&mixer, &projected.narrow(0, 2, 4)?, &mut continued)?;
    assert_eq!(
        fast_state.ssm.flatten_all()?.to_vec1::<f32>()?,
        continued.ssm.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        fast_state.conv.flatten_all()?.to_vec1::<f32>()?,
        continued.conv.flatten_all()?.to_vec1::<f32>()?
    );

    let mut repeated = initial;
    let repeated_output = run(&mixer, &projected, &mut repeated)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual, repeated_output);
    assert_eq!(
        fast_state.ssm.flatten_all()?.to_vec1::<f32>()?,
        repeated.ssm.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn chunked_scan_matches_sequential_scan_and_continues() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    // Four chunks, including a partial final chunk. Nonzero initial state and
    // conv history exercise both the affine prefix and continued prefill.
    let (mixer, projected) = fixture_with_tokens(&device, 8, 769)?;
    let mut initial = mixer.zero_state()?;
    initial.conv = Tensor::from_vec(
        (0..(mixer.kernel - 1) * mixer.conv_dim)
            .map(|i| (i % 23) as f32 * 0.003)
            .collect::<Vec<_>>(),
        (mixer.kernel - 1, mixer.conv_dim),
        &device,
    )?;
    initial.ssm = Tensor::from_vec(
        (0..mixer.heads * mixer.head_dim * mixer.state_dim)
            .map(|i| (i % 31) as f32 * 0.001)
            .collect::<Vec<_>>(),
        (mixer.heads, mixer.head_dim, mixer.state_dim),
        &device,
    )?;
    let mut sequential = initial.clone();
    let expected = run_with_options(&mixer, &projected, &mut sequential, false, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut chunked = initial.clone();
    let actual = run_with_options(&mixer, &projected, &mut chunked, true, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_output = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let expected_state = sequential.ssm.flatten_all()?.to_vec1::<f32>()?;
    let actual_state = chunked.ssm.flatten_all()?.to_vec1::<f32>()?;
    let max_state = expected_state
        .iter()
        .zip(&actual_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_output < 5e-3, "chunked output max abs {max_output}");
    assert!(max_state < 5e-4, "chunked state max abs {max_state}");
    assert_eq!(
        sequential.conv.flatten_all()?.to_vec1::<f32>()?,
        chunked.conv.flatten_all()?.to_vec1::<f32>()?
    );

    let mut continued = initial.clone();
    run_with_options(
        &mixer,
        &projected.narrow(0, 0, 513)?,
        &mut continued,
        true,
        true,
    )?;
    let continued_output = run_with_options(
        &mixer,
        &projected.narrow(0, 513, 256)?,
        &mut continued,
        true,
        true,
    )?
    .flatten_all()?
    .to_vec1::<f32>()?;
    let suffix = &expected[513 * mixer.heads * mixer.head_dim..];
    let continued_max = suffix
        .iter()
        .zip(&continued_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        continued_max < 5e-3,
        "continued output max abs {continued_max}"
    );
    let continued_state = continued.ssm.flatten_all()?.to_vec1::<f32>()?;
    let continued_state_max = expected_state
        .iter()
        .zip(&continued_state)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        continued_state_max < 5e-4,
        "continued state max abs {continued_state_max}"
    );

    let mut repeated = initial;
    let repeat_output = run_with_options(&mixer, &projected, &mut repeated, true, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual, repeat_output);
    assert_eq!(actual_state, repeated.ssm.flatten_all()?.to_vec1::<f32>()?);
    Ok(())
}

#[test]
fn fused_gate_norm_matches_candle_epilogue() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    // Group width 512 exercises Candle's vectorized FP32 reduction, including
    // the same warp reduction order used by the fused kernel.
    let (mixer, projected) = fixture_with_tokens(&device, 64, 513)?;
    let mut old_state = mixer.zero_state()?;
    let mut fused_state = old_state.clone();
    let expected = run_with_options(&mixer, &projected, &mut old_state, true, false)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = run_with_options(&mixer, &projected, &mut fused_state, true, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_abs <= 1e-5, "fused gate/norm max abs {max_abs}");
    assert_eq!(
        old_state.ssm.flatten_all()?.to_vec1::<f32>()?,
        fused_state.ssm.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn fused_decode_step_matches_candle_and_is_deterministic() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => return Ok(()),
    };
    let (mixer, projected) = fixture(&device, 64)?;
    let row = projected.get(0)?;
    let mut initial = mixer.zero_state()?;
    initial.conv = Tensor::from_vec(
        (0..(mixer.kernel - 1) * mixer.conv_dim)
            .map(|i| (i % 23) as f32 * 0.003)
            .collect::<Vec<_>>(),
        (mixer.kernel - 1, mixer.conv_dim),
        &device,
    )?;
    initial.ssm = Tensor::from_vec(
        (0..mixer.heads * mixer.head_dim * mixer.state_dim)
            .map(|i| (i % 31) as f32 * 0.001)
            .collect::<Vec<_>>(),
        (mixer.heads, mixer.head_dim, mixer.state_dim),
        &device,
    )?;
    let mut old = initial.clone();
    let mut fused = initial.clone();
    let expected = mixer
        .step(&row, &mut old)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = run_step(&mixer, &row, &mut fused)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_output = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let old_ssm = old.ssm.flatten_all()?.to_vec1::<f32>()?;
    let fused_ssm = fused.ssm.flatten_all()?.to_vec1::<f32>()?;
    let max_state = old_ssm
        .iter()
        .zip(&fused_ssm)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_output < 5e-3, "decode output max abs {max_output}");
    assert!(max_state < 5e-5, "decode state max abs {max_state}");

    let mut repeated = initial;
    let repeat = run_step(&mixer, &row, &mut repeated)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual, repeat);
    assert_eq!(fused_ssm, repeated.ssm.flatten_all()?.to_vec1::<f32>()?);
    Ok(())
}
