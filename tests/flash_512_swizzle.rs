//! Full-output controls for the opt-in 512-wide shared-memory permutation.
#![cfg(feature = "cuda")]

use anyhow::{ensure, Result};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Storage, Tensor};

fn ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        anyhow::bail!("expected CUDA tensor")
    };
    let base = match t.dtype() {
        DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
        DType::F16 => *storage.as_cuda_slice::<half::f16>()?.device_ptr(),
        DType::U32 => *storage.as_cuda_slice::<u32>()?.device_ptr(),
        _ => anyhow::bail!("unexpected fixture dtype"),
    };
    Ok(base + (layout.start_offset() * t.dtype().size_in_bytes()) as u64)
}

fn words(t: &Tensor) -> Result<Vec<u16>> {
    let flat = t.flatten_all()?;
    Ok(match t.dtype() {
        DType::BF16 => flat
            .to_vec1::<half::bf16>()?
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        DType::F16 => flat
            .to_vec1::<half::f16>()?
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        _ => anyhow::bail!("unexpected output dtype"),
    })
}

#[allow(clippy::too_many_arguments)]
fn check(
    device: &Device,
    dtype: DType,
    lengths: &[(usize, usize)], // (new queries, carried prefix)
    heads: usize,
    kv_heads: usize,
    block_size: usize,
    window: u32,
    softcap: f32,
) -> Result<()> {
    let mut rng = 0x1278_3456u32;
    let mut random = |len| {
        (0..len)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                (rng % 2001) as f32 / 1000. - 1.
            })
            .collect::<Vec<_>>()
    };
    let rows: usize = lengths.iter().map(|&(q, _)| q).sum();
    let stride = lengths
        .iter()
        .map(|&(q, prefix)| (q + prefix).div_ceil(block_size))
        .max()
        .unwrap();
    let blocks = stride * lengths.len();
    // Reverse the physical pages: logical positions must still use the table.
    let table = (0..blocks).rev().map(|p| p as u32).collect::<Vec<_>>();
    let table_gpu = Tensor::from_vec(table.clone(), (lengths.len(), stride), device)?;
    let mut cumulative = vec![0u32];
    for &(q, _) in lengths {
        cumulative.push(cumulative.last().unwrap() + q as u32);
    }
    let cumulative_gpu = Tensor::from_vec(cumulative.clone(), lengths.len() + 1, device)?;
    let contexts = lengths.iter().map(|&(q, p)| (q + p) as u32).collect();
    let contexts_gpu = Tensor::from_vec(contexts, lengths.len(), device)?;
    let query = Tensor::from_vec(random(rows * heads * 512), (rows, heads, 512), device)?
        .to_dtype(dtype)?;
    let key = Tensor::from_vec(
        random(blocks * block_size * kv_heads * 512),
        (blocks, block_size, kv_heads, 512),
        device,
    )?
    .to_dtype(dtype)?;
    let value = Tensor::from_vec(
        random(blocks * block_size * kv_heads * 512),
        key.shape(),
        device,
    )?
    .to_dtype(dtype)?;
    let baseline = Tensor::zeros(query.shape(), dtype, device)?;
    // Kernel must write every output word, including partial query/KV tiles.
    let swizzled = unsafe { Tensor::empty_(query.shape(), dtype, device)? };
    let cuda = device.as_cuda_device()?;
    for (swizzle, output) in [(false, &baseline), (true, &swizzled)] {
        let launch = if swizzle {
            attention_rs::kernels::ffi::call_flash_prefill_paged_512_swizzled
                as unsafe extern "C" fn(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _)
        } else {
            attention_rs::kernels::ffi::call_flash_prefill_paged
        };
        unsafe {
            launch(
                ptr(&query)? as *const _,
                ptr(&key)? as *const _,
                ptr(&value)? as *const _,
                ptr(output)? as *mut _,
                ptr(&table_gpu)? as *const _,
                stride as u32,
                ptr(&cumulative_gpu)? as *const _,
                ptr(&contexts_gpu)? as *const _,
                lengths.len() as u32,
                lengths.iter().map(|&(q, _)| q).max().unwrap() as u32,
                heads as u32,
                kv_heads as u32,
                512,
                block_size as u32,
                window,
                1,
                1. / 512f32.sqrt(),
                softcap,
                i32::from(dtype == DType::BF16),
                *cuda.cu_stream() as i64,
            );
        }
    }
    device.synchronize()?;
    ensure!(
        words(&baseline)? == words(&swizzled)?,
        "output word mismatch"
    );

    // Independent FP64 softmax oracle for every row/head on short fixtures.
    if rows <= 72 {
        let q = query
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let k = key.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v = value
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let y = swizzled
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_abs = 0f64;
        for (seq, &(len, prefix)) in lengths.iter().enumerate() {
            for row in 0..len {
                let logical = prefix + row;
                let start = if window == 0 {
                    0
                } else {
                    (logical + 1).saturating_sub(window as usize)
                };
                for head in 0..heads {
                    let base = ((cumulative[seq] as usize + row) * heads + head) * 512;
                    let kv_head = head / (heads / kv_heads);
                    let kv_base = |token: usize| {
                        let physical = table[seq * stride + token / block_size] as usize;
                        ((physical * block_size + token % block_size) * kv_heads + kv_head) * 512
                    };
                    let scores: Vec<f64> = (start..=logical)
                        .map(|token| {
                            let kb = kv_base(token);
                            let score = (0..512)
                                .map(|d| q[base + d] as f64 * k[kb + d] as f64)
                                .sum::<f64>()
                                * (1. / 512f32.sqrt()) as f64;
                            if softcap > 0. {
                                softcap as f64 * (score / softcap as f64).tanh()
                            } else {
                                score
                            }
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let p: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                    let z: f64 = p.iter().sum();
                    for d in 0..512 {
                        let expected: f64 = p
                            .iter()
                            .enumerate()
                            .map(|(i, p)| p * v[kv_base(start + i) + d] as f64 / z)
                            .sum();
                        let actual = y[base + d] as f64;
                        ensure!(actual.is_finite());
                        max_abs = max_abs.max((expected - actual).abs());
                    }
                }
            }
        }
        eprintln!("flash512_oracle dtype={dtype:?} lengths={lengths:?} window={window} softcap={softcap} max_abs={max_abs}");
        // BF16 probabilities and output rounding are intentional in both paths.
        ensure!(max_abs < 0.009, "CPU attention oracle mismatch: {max_abs}");
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM80+ CUDA GPU with at least 101120 bytes shared memory per block"]
fn swizzled_512_matches_original_and_cpu() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16] {
        check(&device, dtype, &[(1, 0)], 1, 1, 16, 0, 0.)?;
        check(&device, dtype, &[(33, 7), (39, 19)], 8, 2, 16, 0, 0.)?;
        check(&device, dtype, &[(17, 63), (35, 3)], 8, 2, 64, 23, 2.3)?;
        check(&device, dtype, &[(2051, 37)], 8, 2, 64, 0, 0.)?;
    }
    Ok(())
}
