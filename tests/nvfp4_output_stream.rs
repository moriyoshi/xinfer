//! Opt-in full-output controls for native CUTLASS and stream interoperability.
#![cfg(all(feature = "cuda", feature = "cutlass"))]

use anyhow::{ensure, Result};
use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::{DType, Device, Storage, Tensor};

fn ptr(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        anyhow::bail!("expected CUDA tensor")
    };
    let base = match tensor.dtype() {
        DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
        DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
        DType::U8 => *storage.as_cuda_slice::<u8>()?.device_ptr(),
        _ => anyhow::bail!("unsupported fixture dtype"),
    };
    Ok(base + (layout.start_offset() * tensor.dtype().size_in_bytes()) as u64)
}

#[test]
#[ignore = "requires native Blackwell NVFP4; empty mode is an explicit CUDA initcheck diagnostic"]
fn native_fp4_output_coverage() -> Result<()> {
    let stream_mode = std::env::var("NVFP4_COVERAGE_STREAM").unwrap_or_else(|_| "explicit".into());
    let init = std::env::var("NVFP4_COVERAGE_INIT").unwrap_or_else(|_| "ones".into());
    let fence = std::env::var("NVFP4_COVERAGE_FENCE").unwrap_or_else(|_| "stream".into());
    let prepare_fence = std::env::var("NVFP4_COVERAGE_PRE_FENCE").unwrap_or_else(|_| "none".into());
    let device = match stream_mode.as_str() {
        "explicit" => Device::new_cuda_with_stream(0)?,
        "legacy" => Device::new_cuda(0)?,
        _ => anyhow::bail!("unknown stream mode"),
    };
    let cuda = device.as_cuda_device()?;
    ensure!(attention_rs::cuda_utils::sm_version(cuda).unwrap_or(0) >= 120);
    ensure!(["stream", "context"].contains(&fence.as_str()));
    let random = std::env::var("NVFP4_COVERAGE_RANDOM").as_deref() == Ok("1");
    let mut rng = 0x72616e646f6d_u64;
    let mut bytes = |len: usize, scales: bool| -> Vec<u8> {
        (0..len)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if scales {
                    0x28 + (rng % 32) as u8
                } else {
                    rng as u8
                }
            })
            .collect()
    };
    for (m, n, k) in [
        (3usize, 160usize, 128usize),
        (1, 128, 128),
        (127, 160, 256),
        (128, 160, 256),
        (129, 160, 256),
        (2048, 10240, 2560),
        (8192, 20480, 2560),
    ]
    .into_iter()
    .filter(|(m, _, _)| std::env::var("NVFP4_COVERAGE_SMALL").as_deref() != Ok("1") || *m <= 129)
    {
        // FP4 code 2 and E4M3 code 0x38 both represent +1 exactly. Every
        // result must therefore equal K, independent of accumulation order.
        let input = Tensor::from_vec(
            if random {
                bytes(m * k / 2, false)
            } else {
                vec![0x22u8; m * k / 2]
            },
            (m, k / 2),
            &device,
        )?;
        let weight = Tensor::from_vec(
            if random {
                bytes(n * k / 2, false)
            } else {
                vec![0x22u8; n * k / 2]
            },
            (n, k / 2),
            &device,
        )?;
        let scale_cols = (k / 16).div_ceil(4) * 4;
        let input_sf = Tensor::from_vec(
            if random {
                bytes(m.div_ceil(128) * 128 * scale_cols, true)
            } else {
                vec![0x38u8; m.div_ceil(128) * 128 * scale_cols]
            },
            (m.div_ceil(128) * 128, scale_cols),
            &device,
        )?;
        let weight_sf = Tensor::from_vec(
            if random {
                bytes(n.div_ceil(128) * 128 * scale_cols, true)
            } else {
                vec![0x38u8; n.div_ceil(128) * 128 * scale_cols]
            },
            (n.div_ceil(128) * 128, scale_cols),
            &device,
        )?;
        let alpha = Tensor::from_vec(vec![1f32], 1, &device)?;
        let mut reference: Option<Vec<half::bf16>> = None;
        let modes: &[&str] = if random {
            &["zeros", "ones", "empty", "empty"]
        } else {
            &[init.as_str()]
        };
        for init in modes {
            let output = match *init {
                "zeros" => Tensor::zeros((m, n), DType::BF16, &device)?,
                "ones" => Tensor::ones((m, n), DType::BF16, &device)?,
                // This ignored diagnostic deliberately checks a potentially
                // incomplete writer. It is never used by normal inference.
                "empty" => unsafe { Tensor::empty_((m, n), DType::BF16, &device)? },
                _ => anyhow::bail!("unknown initialization mode"),
            };
            let (workspace, bytes) = attention_rs::workspace::get_cutlass_workspace(cuda, 0)?;
            // Optional control separating allocation readiness from write coverage.
            match prepare_fence.as_str() {
                "context" => unsafe {
                    sys::lib().cuCtxSynchronize().result()?;
                },
                "stream" => device.synchronize()?,
                "none" => (),
                _ => anyhow::bail!("unknown preparation fence"),
            }
            let status = unsafe {
                attention_rs::kernels::ffi::nvfp4_cutlass_gemm_bf16_checked(
                    ptr(&input)? as *const _,
                    ptr(&weight)? as *const _,
                    ptr(&input_sf)? as *const _,
                    ptr(&weight_sf)? as *const _,
                    ptr(&alpha)? as *const _,
                    ptr(&output)? as *mut _,
                    m as i32,
                    n as i32,
                    k as i32,
                    workspace,
                    bytes as i64,
                    *cuda.cu_stream() as i64,
                )
            };
            ensure!(status == 0, "native launch failed: {status}");
            if fence == "context" {
                unsafe {
                    sys::lib().cuCtxSynchronize().result()?;
                }
            } else {
                device.synchronize()?;
            }
            let expected = half::bf16::from_f32(k as f32).to_bits();
            let words = output.flatten_all()?.to_vec1::<half::bf16>()?;
            let bad = if random {
                ensure!(words.iter().all(|v| v.is_finite()), "nonfinite output");
                match &reference {
                    Some(expected) => words
                        .iter()
                        .zip(expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count(),
                    None => {
                        reference = Some(words.clone());
                        0
                    }
                }
            } else {
                words.iter().filter(|v| v.to_bits() != expected).count()
            };
            eprintln!("nvfp4_output_coverage stream={stream_mode} init={init} pre_fence={prepare_fence} fence={fence} m={m} n={n} k={k} words={} bad={bad}",words.len());
            ensure!(
                bad == 0,
                "native GEMM did not produce every expected output word (random={random})"
            );
        }
    }
    Ok(())
}

/// Full quantization + swizzling + epilogue control. Run alone: environment
/// switches are process-global and this ignored diagnostic owns them.
#[test]
#[ignore = "requires native Blackwell NVFP4 and the prefill-overwrite experiment"]
fn native_fp4_prefill_overwrite_matches_initialized_pipeline() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    ensure!(attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0) >= 120);
    let mut rng = 0x243f6a8885a308d3u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for (m, n, k) in [
        (3, 160, 128),
        (127, 160, 256),
        (128, 160, 256),
        (129, 160, 256),
        (2048, 10240, 2560),
        (8192, 20480, 2560),
    ]
    .into_iter()
    .filter(|(m, _, _)| std::env::var("NVFP4_COVERAGE_SMALL").as_deref() != Ok("1") || *m <= 129)
    {
        let input = Tensor::from_vec(
            (0..m * k)
                .map(|_| half::bf16::from_f32((next() % 2001) as f32 / 1000. - 1.))
                .collect::<Vec<_>>(),
            (m, k),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..n * k / 2).map(|_| next() as u8).collect::<Vec<_>>(),
            (n, k / 2),
            &device,
        )?;
        let scale = Tensor::from_vec(
            (0..n * k / 16)
                .map(|_| 0x28 + (next() % 32) as u8)
                .collect::<Vec<_>>(),
            (n, k / 16),
            &device,
        )?;
        let bias = Tensor::from_vec(
            (0..n)
                .map(|_| half::bf16::from_f32((next() % 201) as f32 / 100. - 1.))
                .collect::<Vec<_>>(),
            n,
            &device,
        )?;
        let swizzled = attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(&scale)?;
        for resident in [false, true] {
            for with_bias in [false, true] {
                let run = |enabled: bool| -> Result<Vec<half::bf16>> {
                    std::env::set_var(
                        "XINFER_NVFP4_PREFILL_OVERWRITE",
                        if enabled { "1" } else { "0" },
                    );
                    let out = attention_rs::nvfp4_linear::nvfp4_matmul(
                        &input,
                        &weight,
                        &scale,
                        0.25,
                        0.5,
                        if with_bias { Some(&bias) } else { None },
                        true,
                        if resident { Some(&swizzled) } else { None },
                        None,
                    )?;
                    Ok(out.flatten_all()?.to_vec1::<half::bf16>()?)
                };
                let reference = run(false)?;
                ensure!(reference.iter().all(|v| v.is_finite()));
                for _ in 0..2 {
                    let actual = run(true)?;
                    let bad = reference
                        .iter()
                        .zip(&actual)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    eprintln!("native_fp4_pipeline m={m} n={n} k={k} resident={resident} bias={with_bias} bad={bad}");
                    ensure!(bad == 0, "initialized/overwrite full pipeline differs");
                }
            }
        }
    }
    std::env::remove_var("XINFER_NVFP4_PREFILL_OVERWRITE");
    Ok(())
}

#[test]
fn checked_native_output_rejects_invalid_problem_before_launch() {
    let status = unsafe {
        attention_rs::kernels::ffi::nvfp4_cutlass_gemm_bf16_checked(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            32,
            32,
            std::ptr::null_mut(),
            0,
            0,
        )
    };
    assert_ne!(status, 0);
}
