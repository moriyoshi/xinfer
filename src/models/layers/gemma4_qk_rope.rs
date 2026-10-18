//! F32 Q/K RMSNorm and RoPE fused into BF16-output CUDA kernels for Gemma 4.

use candle_core::cuda_backend::cudarc::driver::{capture_status, sys::CUstreamCaptureStatus};
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchAsync, LaunchConfig};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::{DType, Result, Storage, Tensor};
use std::sync::Mutex;

const KERNEL: &str = "xinfer_gemma4_qk_norm_rope";
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

fn ptr(t: &Tensor, dtype: DType) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("Gemma Q/K RoPE requires CUDA tensors")
    };
    let base = match dtype {
        DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
        DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
        DType::I64 => *storage.as_cuda_slice::<i64>()?.device_ptr(),
        _ => candle_core::bail!("unsupported Gemma Q/K RoPE dtype"),
    };
    Ok(base + (layout.start_offset() * dtype.size_in_bytes()) as u64)
}

fn module(device: &candle_core::CudaDevice) -> Result<String> {
    let cuda = device.cuda_device();
    let sm = attention_rs::cuda_utils::sm_version(device).ok_or_else(|| {
        candle_core::Error::Msg("unknown CUDA architecture for Gemma Q/K RoPE".into())
    })?;
    if sm < 80 {
        candle_core::bail!("Gemma Q/K RoPE requires compute capability 8.0 or later")
    }
    let name = format!("xinfer_gemma4_qk_rope_sm{sm}_v2");
    let _guard = COMPILE_LOCK.lock().unwrap();
    if !cuda.has_func(&name, KERNEL) {
        let capture = capture_status(*cuda.cu_stream())
            .map_err(|e| candle_core::Error::Msg(format!("check Gemma Q/K RoPE capture: {e}")))?;
        if capture == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            candle_core::bail!("Gemma Q/K RoPE must be loaded before CUDA graph capture")
        }
        let ptx = compile_ptx_with_opts(
            include_str!("gemma4_qk_rope.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{sm}")],
                ..Default::default()
            },
        )
        .map_err(|e| candle_core::Error::Msg(format!("compile Gemma Q/K RoPE: {e}")))?;
        cuda.load_ptx(ptx, &name, &[KERNEL])
            .map_err(|e| candle_core::Error::Msg(format!("load Gemma Q/K RoPE: {e}")))?;
    }
    Ok(name)
}

pub(crate) fn prepare(device: &candle_core::Device) -> Result<()> {
    module(device.as_cuda_device()?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    q: &Tensor,
    k: Option<(&Tensor, Tensor, f32)>,
    q_weight: &Tensor,
    q_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    interleaved: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    if !q.device().is_cuda()
        || q.dtype() != DType::BF16
        || q_weight.dtype() != DType::F32
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || positions.dtype() != DType::I64
        || ![q_weight, cos, sin, positions]
            .iter()
            .all(|t| t.is_contiguous() && t.device().same_device(q.device()))
        || q.rank() != 3
        || q_weight.rank() != 1
        || cos.rank() != 2
        || cos.shape() != sin.shape()
        || positions.rank() != 1
    {
        return Ok(None);
    }
    let (tokens, q_heads, width) = q.dims3()?;
    let (q_storage, q_layout) = q.storage_and_layout();
    let q_stride = q_layout.stride().to_vec();
    drop(q_storage);
    if q_stride.len() != 3
        || q_stride[1] != width
        || q_stride[2] != 1
        || q_stride[0] < q_heads * width
    {
        return Ok(None);
    }
    let (k_heads, k_token_stride) = if let Some((k, weight, _)) = &k {
        if k.dtype() != DType::BF16
            || weight.dtype() != DType::F32
            || !k.device().same_device(q.device())
            || !weight.device().same_device(q.device())
            || !weight.is_contiguous()
            || k.rank() != 3
            || k.dim(0)? != tokens
            || k.dim(2)? != width
            || weight.dims() != [width]
        {
            return Ok(None);
        }
        let heads = k.dim(1)?;
        let (k_storage, k_layout) = k.storage_and_layout();
        let stride = k_layout.stride().to_vec();
        drop(k_storage);
        if stride.len() != 3 || stride[1] != width || stride[2] != 1 || stride[0] < heads * width {
            return Ok(None);
        }
        (heads, stride[0])
    } else {
        (0, 0)
    };
    if tokens == 0
        || tokens > u32::MAX as usize
        || q_heads == 0
        || q_heads + k_heads > 65_535
        || width < 32
        || width > 1024
        || width % 2 != 0
        || q_weight.dims() != [width]
        || cos.dim(1)? != width / 2
        || positions.dims() != [tokens]
    {
        return Ok(None);
    }
    let device = q.device().as_cuda_device()?;
    let cuda = device.cuda_device();
    let name = module(device)?;
    let output_q = unsafe { Tensor::empty_(q.shape().clone(), DType::BF16, q.device())? };
    let output_k = if let Some((k, _, _)) = &k {
        Some(unsafe { Tensor::empty_(k.shape().clone(), DType::BF16, q.device())? })
    } else {
        None
    };
    let function = cuda
        .get_func(&name, KERNEL)
        .ok_or_else(|| candle_core::Error::Msg("missing Gemma Q/K RoPE kernel".into()))?;
    let (k_ptr, k_weight_ptr, k_out_ptr, k_eps) = if let Some((k, weight, eps)) = &k {
        (
            ptr(k, DType::BF16)?,
            ptr(weight, DType::F32)?,
            ptr(output_k.as_ref().unwrap(), DType::BF16)?,
            *eps,
        )
    } else {
        (0, 0, 0, 0.0)
    };
    unsafe {
        function.launch(
            LaunchConfig {
                grid_dim: (tokens as u32, (q_heads + k_heads) as u32, 1),
                // Match Candle's F32 RMSNorm reduction order exactly.
                block_dim: (if width < 1024 { 32 } else { 1024 }, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                ptr(q, DType::BF16)?,
                k_ptr,
                ptr(q_weight, DType::F32)?,
                k_weight_ptr,
                ptr(cos, DType::F32)?,
                ptr(sin, DType::F32)?,
                ptr(positions, DType::I64)?,
                ptr(&output_q, DType::BF16)?,
                k_out_ptr,
                q_heads as i32,
                k_heads as i32,
                width as i32,
                q_stride[0] as u64,
                k_token_stride as u64,
                q_eps,
                k_eps,
                interleaved as i32,
            ),
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("launch Gemma Q/K RoPE: {e}")))?;
    Ok(Some((output_q, output_k)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use attention_rs::fused_rope::FusedRope;
    use std::time::Instant;

    #[test]
    fn packed_qk_row_strides_match_contiguous_inputs() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        let (tokens, q_heads, k_heads, width) = (7, 8, 2, 256);
        let q_cols = q_heads * width;
        let k_cols = k_heads * width;
        let packed_cols = q_cols + 2 * k_cols;
        let packed = Tensor::from_vec(
            (0..tokens * packed_cols)
                .map(|i| ((i * 53 % 401) as f32 - 200.0) / 59.0)
                .collect::<Vec<_>>(),
            (tokens, packed_cols),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let packed_heads = packed.reshape((tokens, packed_cols / width, width))?;
        let q = packed_heads.narrow(1, 0, q_heads)?;
        let k = packed_heads.narrow(1, q_heads, k_heads)?;
        assert!(!q.is_contiguous() && !k.is_contiguous());
        let weight = Tensor::ones((width,), DType::F32, &device)?;
        let cos = Tensor::ones((1, width / 2), DType::F32, &device)?;
        let sin = Tensor::zeros((1, width / 2), DType::F32, &device)?;
        let positions = Tensor::zeros((tokens,), DType::I64, &device)?;
        let run = |q: &Tensor, k: &Tensor| -> Result<(Tensor, Tensor)> {
            let (q_out, k_out) = forward(
                q,
                Some((k, weight.clone(), 1e-6)),
                &weight,
                1e-6,
                &cos,
                &sin,
                &positions,
                true,
            )?
            .expect("supported Q/K layout");
            Ok((q_out, k_out.expect("key output")))
        };
        let (q_view, k_view) = run(&q, &k)?;
        let (q_ref, k_ref) = run(&q.contiguous()?, &k.contiguous()?)?;
        for (actual, expected) in [(q_view, q_ref), (k_view, k_ref)] {
            let actual = actual.flatten_all()?.to_vec1::<half::bf16>()?;
            let expected = expected.flatten_all()?.to_vec1::<half::bf16>()?;
            assert_eq!(actual.len(), expected.len());
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        Ok(())
    }

    #[test]
    fn fused_qk_matches_f32_norm_then_rope() -> Result<()> {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            return Ok(());
        };
        if attention_rs::cuda_utils::sm_version(device.as_cuda_device()?).unwrap_or(0) < 80 {
            return Ok(());
        }
        for (tokens, q_heads, k_heads, width, interleaved) in
            [(7, 8, 2, 256, true), (19, 4, 1, 128, false)]
        {
            let make_input = |heads| -> Result<Tensor> {
                let values = (0..tokens * heads * width)
                    .map(|i| ((i * 73 % 499) as f32 - 249.0) / 67.0)
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, (tokens, heads, width), &device)?.to_dtype(DType::BF16)
            };
            let q = make_input(q_heads)?;
            let k = make_input(k_heads)?;
            let q_weight = Tensor::from_vec(
                (0..width)
                    .map(|i| 0.5 + i as f32 / width as f32)
                    .collect::<Vec<_>>(),
                width,
                &device,
            )?;
            let k_weight = Tensor::from_vec(
                (0..width)
                    .map(|i| 0.75 + i as f32 / (2 * width) as f32)
                    .collect::<Vec<_>>(),
                width,
                &device,
            )?;
            let table = |f: fn(f32) -> f32| -> Result<Tensor> {
                Tensor::from_vec(
                    (0..32 * width / 2)
                        .map(|i| f((i / (width / 2)) as f32 * (i % (width / 2) + 1) as f32 / 73.0))
                        .collect::<Vec<_>>(),
                    (32, width / 2),
                    &device,
                )
            };
            let cos = table(f32::cos)?;
            let sin = table(f32::sin)?;
            let positions = Tensor::from_vec(
                (0..tokens)
                    .map(|i| ((i * 3 + 2) % 31) as i64)
                    .collect::<Vec<_>>(),
                tokens,
                &device,
            )?;
            let q_ref = candle_nn::ops::rms_norm(&q.to_dtype(DType::F32)?, &q_weight, 1e-6)?;
            let k_ref = candle_nn::ops::rms_norm(&k.to_dtype(DType::F32)?, &k_weight, 1e-6)?;
            FusedRope::apply_inplace(&q_ref, &k_ref, &cos, &sin, &positions, interleaved)?;
            let q_ref = q_ref.to_dtype(DType::BF16)?;
            let k_ref = k_ref.to_dtype(DType::BF16)?;
            let (q_new, k_new) = forward(
                &q,
                Some((&k, k_weight, 1e-6)),
                &q_weight,
                1e-6,
                &cos,
                &sin,
                &positions,
                interleaved,
            )?
            .expect("supported Gemma Q/K kernel inputs");
            let k_new = k_new.expect("key output");
            for (expected, actual) in [(q_ref, q_new), (k_ref, k_new)] {
                let expected = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let max_abs = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert_eq!(max_abs, 0.0, "width={width} interleaved={interleaved}");
            }
            let (q_only, no_key) = forward(
                &q,
                None,
                &q_weight,
                1e-6,
                &cos,
                &sin,
                &positions,
                interleaved,
            )?
            .expect("supported query-only inputs");
            assert!(no_key.is_none());
            let q_only = q_only
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let q_full = forward(
                &q,
                Some((&k, Tensor::ones(width, DType::F32, &device)?, 1e-6)),
                &q_weight,
                1e-6,
                &cos,
                &sin,
                &positions,
                interleaved,
            )?
            .unwrap()
            .0
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            assert_eq!(q_only, q_full, "query-only and Q/K output must agree");
        }
        Ok(())
    }

    #[test]
    #[ignore = "GPU timing benchmark; run only on an idle GPU"]
    fn bench_qk_norm_rope_8192() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let (tokens, q_heads, k_heads, width) = (8192, 8, 2, 256);
        let q = Tensor::from_vec(
            (0..tokens * q_heads * width)
                .map(|i| ((i * 73 % 499) as f32 - 249.0) / 67.0)
                .collect::<Vec<_>>(),
            (tokens, q_heads, width),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let k = Tensor::from_vec(
            (0..tokens * k_heads * width)
                .map(|i| ((i * 47 % 391) as f32 - 195.0) / 71.0)
                .collect::<Vec<_>>(),
            (tokens, k_heads, width),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weight = Tensor::ones(width, DType::F32, &device)?;
        let cos = Tensor::ones((tokens, width / 2), DType::F32, &device)?;
        let sin = Tensor::zeros((tokens, width / 2), DType::F32, &device)?;
        let positions = Tensor::arange(0_i64, tokens as i64, &device)?;
        let baseline = || -> Result<()> {
            let q_ref = candle_nn::ops::rms_norm(&q.to_dtype(DType::F32)?, &weight, 1e-6)?;
            let k_ref = candle_nn::ops::rms_norm(&k.to_dtype(DType::F32)?, &weight, 1e-6)?;
            FusedRope::apply_inplace(&q_ref, &k_ref, &cos, &sin, &positions, false)?;
            let _ = q_ref.to_dtype(DType::BF16)?;
            let _ = k_ref.to_dtype(DType::BF16)?;
            device.synchronize()?;
            Ok(())
        };
        let fused = || -> Result<()> {
            let _ = forward(
                &q,
                Some((&k, weight.clone(), 1e-6)),
                &weight,
                1e-6,
                &cos,
                &sin,
                &positions,
                false,
            )?
            .expect("supported benchmark shape");
            device.synchronize()?;
            Ok(())
        };
        baseline()?;
        fused()?;
        let mut baseline_ms = Vec::new();
        let mut fused_ms = Vec::new();
        for sample in 0..20 {
            for (kind, timings) in if sample % 2 == 0 {
                [(0, &mut baseline_ms), (1, &mut fused_ms)]
            } else {
                [(1, &mut fused_ms), (0, &mut baseline_ms)]
            } {
                let started = Instant::now();
                if kind == 0 {
                    baseline()?
                } else {
                    fused()?
                }
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        baseline_ms.sort_by(f64::total_cmp);
        fused_ms.sort_by(f64::total_cmp);
        eprintln!("gemma4_qk_8192 baseline_p50_ms={:.3} fused_p50_ms={:.3} baseline_p95_ms={:.3} fused_p95_ms={:.3}",
            baseline_ms[10], fused_ms[10], baseline_ms[19], fused_ms[19]);
        Ok(())
    }
}
