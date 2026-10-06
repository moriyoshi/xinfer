//! Fixtures were produced by shifou's compact-KV encoder and independent CPU
//! decoder for a 96-token, 2-head, 16-channel tile with a 32-token exact tail.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use xinfer::utils::packed_kv_restore::{
    restore_dense_packed_kv_pages, DensePackedKvPage, DensePackedKvRestore, PackedKvAxis,
    PackedKvException,
};
use xinfer::utils::GpuKvCache;

fn fixture(axis: PackedKvAxis, bits: u8) -> Result<(DensePackedKvPage, Vec<u16>)> {
    let stem = format!(
        "{}-{bits}",
        if axis == PackedKvAxis::Key {
            "token"
        } else {
            "channel"
        }
    );
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/packed_kv");
    let read = |suffix: &str| std::fs::read(dir.join(format!("{stem}-{suffix}.bin")));
    let codes = read("codes")?;
    let params = read("parameters")?
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let tail = read("tail")?
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let expected = read("expected")?
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let group_size = if axis == PackedKvAxis::Key { 64 } else { 16 };
    let page = DensePackedKvPage::new(
        axis,
        bits,
        96,
        2,
        16,
        group_size,
        32,
        codes,
        params,
        tail,
        vec![],
    )?;
    Ok((page, expected))
}

#[test]
fn shifou_compact_fixtures_match_cpu_oracle() -> Result<()> {
    for axis in [PackedKvAxis::Key, PackedKvAxis::Value] {
        for bits in [2, 4] {
            let (page, expected) = fixture(axis, bits)?;
            assert_eq!(page.decode_cpu_bf16()?, expected);
            let bytes = page.to_bytes()?;
            assert_eq!(
                DensePackedKvPage::from_bytes(&bytes)?.decode_cpu_bf16()?,
                expected
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn batched_gpu_restore_matches_independent_bf16_and_exact_exceptions() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    let mut pages = Vec::new();
    let mut expected = Vec::new();
    for bits in [2, 4] {
        for axis in [PackedKvAxis::Key, PackedKvAxis::Value] {
            let (page, words) = fixture(axis, bits)?;
            pages.push(page);
            expected.push(words);
        }
    }
    let base = pages[0].clone();
    let exception_page = DensePackedKvPage::new(
        base.axis,
        base.bits,
        base.tokens,
        base.heads,
        base.channels,
        base.group_size,
        base.tail_tokens,
        base.codes,
        base.params,
        base.tail_bf16,
        vec![PackedKvException {
            index: 123,
            bf16_bits: 0x3f80,
        }],
    )?;
    let mut exact_tail = Vec::new();
    for i in 0..(16 * 2 * 16) {
        exact_tail.push((0x3f00 + i % 256) as u16);
    }
    let exact_page = DensePackedKvPage::new(
        PackedKvAxis::Key,
        2,
        16,
        2,
        16,
        16,
        16,
        vec![],
        vec![],
        exact_tail,
        vec![],
    )?;
    pages.push(exception_page);
    pages.push(exact_page);
    let pairs = (0..3)
        .map(|_| {
            Ok((
                Tensor::zeros((2, 96, 2, 16), DType::BF16, &device)?,
                Tensor::zeros((2, 96, 2, 16), DType::BF16, &device)?,
            ))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    let cache = GpuKvCache::Flash(pairs);
    let requests = vec![
        DensePackedKvRestore {
            layer: 0,
            block: 1,
            token_offset: 0,
            page: &pages[0],
        },
        DensePackedKvRestore {
            layer: 0,
            block: 1,
            token_offset: 0,
            page: &pages[1],
        },
        DensePackedKvRestore {
            layer: 1,
            block: 1,
            token_offset: 0,
            page: &pages[2],
        },
        DensePackedKvRestore {
            layer: 1,
            block: 1,
            token_offset: 0,
            page: &pages[3],
        },
        DensePackedKvRestore {
            layer: 2,
            block: 1,
            token_offset: 0,
            page: &pages[4],
        },
        DensePackedKvRestore {
            layer: 2,
            block: 0,
            token_offset: 32,
            page: &pages[5],
        },
    ];
    let stats = restore_dense_packed_kv_pages(&cache, &requests)?;
    assert_eq!(stats.pages, 6);
    assert_eq!(stats.kernel_launches, 1);
    let pairs = cache.as_pairs().unwrap();
    for (index, words) in expected.iter().enumerate() {
        let target = if index % 2 == 0 {
            &pairs[index / 2].0
        } else {
            &pairs[index / 2].1
        };
        let actual = target
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<half::bf16>()?;
        assert!(actual[..3072].iter().all(|word| word.to_bits() == 0));
        assert_eq!(
            actual[3072..]
                .iter()
                .map(|word| word.to_bits())
                .collect::<Vec<_>>(),
            *words
        );
    }
    let exception = pairs[2]
        .0
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<half::bf16>()?;
    assert_eq!(exception[3072 + 123].to_bits(), 0x3f80);
    let oracle = pages[4].decode_cpu_bf16()?;
    assert_eq!(
        exception[3072..]
            .iter()
            .map(|word| word.to_bits())
            .collect::<Vec<_>>(),
        oracle
    );
    let exact = pages[5].decode_cpu_bf16()?;
    assert_eq!(
        exception[32 * 32..48 * 32]
            .iter()
            .map(|word| word.to_bits())
            .collect::<Vec<_>>(),
        exact
    );

    let mut corrupt = pages[0].clone();
    corrupt.codes[0] ^= 1;
    let rejected = [DensePackedKvRestore {
        layer: 0,
        block: 0,
        token_offset: 0,
        page: &corrupt,
    }];
    assert!(restore_dense_packed_kv_pages(&cache, &rejected).is_err());
    let overlapping = [
        DensePackedKvRestore {
            layer: 0,
            block: 0,
            token_offset: 0,
            page: &pages[0],
        },
        DensePackedKvRestore {
            layer: 0,
            block: 0,
            token_offset: 0,
            page: &pages[0],
        },
    ];
    assert!(restore_dense_packed_kv_pages(&cache, &overlapping).is_err());
    let unchanged = pairs[0]
        .0
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<half::bf16>()?;
    assert!(unchanged[..3072].iter().all(|word| word.to_bits() == 0));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn one_launch_restores_128_small_pages() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    let page = DensePackedKvPage::new(
        PackedKvAxis::Key,
        2,
        16,
        2,
        16,
        12,
        4,
        vec![0x39; 12 * 2 * 16 * 2 / 8],
        [0.25f32, 0.125].repeat(2 * 16),
        vec![0x3f80; 4 * 2 * 16],
        vec![],
    )?;
    let expected = page.decode_cpu_bf16()?;
    let cache = GpuKvCache::Flash(vec![(
        Tensor::zeros((128, 16, 2, 16), DType::BF16, &device)?,
        Tensor::zeros((128, 16, 2, 16), DType::BF16, &device)?,
    )]);
    let requests = (0..128)
        .map(|block| DensePackedKvRestore {
            layer: 0,
            block,
            token_offset: 0,
            page: &page,
        })
        .collect::<Vec<_>>();
    let stats = restore_dense_packed_kv_pages(&cache, &requests)?;
    assert_eq!(stats.pages, 128);
    assert_eq!(stats.kernel_launches, 1);
    let actual = cache.as_pairs().unwrap()[0]
        .0
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<half::bf16>()?;
    for block in actual.chunks_exact(expected.len()) {
        assert_eq!(
            block.iter().map(|word| word.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "isolated GB10 packed-upload/expansion timing probe"]
fn qwen3_4b_shaped_restore_timing() -> Result<()> {
    let Ok(device) = Device::new_cuda(0) else {
        return Ok(());
    };
    let cache = GpuKvCache::Flash(vec![(
        Tensor::zeros((1, 7000, 8, 128), DType::BF16, &device)?,
        Tensor::zeros((1, 7000, 8, 128), DType::BF16, &device)?,
    )]);
    let old = 7000 - 32;
    let old_words = old * 8 * 128;
    for axis in [PackedKvAxis::Key, PackedKvAxis::Value] {
        for bits in [2, 4] {
            let groups = if axis == PackedKvAxis::Key {
                8 * 128
            } else {
                old * 8
            };
            let page = DensePackedKvPage::new(
                axis,
                bits,
                7000,
                8,
                128,
                if axis == PackedKvAxis::Key {
                    old as u32
                } else {
                    128
                },
                32,
                vec![0x39; old_words * bits as usize / 8],
                [0.0f32, 0.125].repeat(groups),
                vec![0x3f80; 32 * 8 * 128],
                vec![],
            )?;
            let request = [DensePackedKvRestore {
                layer: 0,
                block: 0,
                token_offset: 0,
                page: &page,
            }];
            let mut samples = Vec::new();
            for iteration in 0..17 {
                let started = std::time::Instant::now();
                restore_dense_packed_kv_pages(&cache, &request)?;
                if iteration >= 2 {
                    samples.push(started.elapsed().as_secs_f64() * 1000.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!("xinfer packed KV restore: axis={axis:?} bits={bits} p50_ms={:.3} p95_ms={:.3} packed_bytes={} output_bytes={}", samples[7], samples[14], page.codes.len() + page.params.len() * 4 + page.tail_bf16.len() * 2, 7000 * 8 * 128 * 2);
        }
    }
    Ok(())
}
