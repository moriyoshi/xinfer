use super::*;
use crate::models::layers::distributed::ReplicatedLinear;
use candle_core::Device;

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
