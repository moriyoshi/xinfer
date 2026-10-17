# Gemma 4 E4B BF16 decode GEMV on GB10

The E4B BF16 decoder spends most of its GPU time reading projection weights for single-token matrix-vector products. An Nsight Systems trace of the real-checkpoint test on a GB10 recorded 4,874 ms of GPU kernels, including 3,793 ms in cuBLAS BF16 GEMV. Paged attention decode contributed 12 ms across the trace. The 20,480 × 2,560 merged gate/up projection was a repeated expensive shape: 2,688 launches averaged 0.691 ms under the profiler. The trace covers several prefills and decode sequences, so these totals are not per-token times.

`XINFER_GEMMA4_BF16_GEMV=1` enables an experimental row-parallel BF16 GEMV during Gemma 4 single-token decode. Each warp computes one output row with FP32 accumulation, and one CUDA kernel writes all BF16 output rows. The mode applies only to contiguous, bias-free BF16 projections on CUDA devices of compute capability 8.0 or newer; other projections use the existing path. It is disabled by default. The kernel is compiled and loaded at model construction, before CUDA graph capture. It does not change prefill or the K/V cache layout.

## Real-checkpoint measurements

Checkpoint: `google/gemma-4-E4B-it` revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, BF16. Build: release with `cuda,cutlass`. The test synchronized after each forward. Samples ran on one GB10; the host is shared, so these are local comparisons rather than serving throughput claims.

| Workload | Existing cuBLAS GEMV | Opt-in GEMV |
| --- | ---: | ---: |
| 40-token prefix, 48 greedy decode steps | 63.99, 65.00, 64.62 ms/token | 52.44, 52.36, 56.05, 51.74 ms/token |
| 640-token prefix, 64 greedy decode steps | 66.88 ms/token | 54.87, 54.58 ms/token |
| Whole differential test, profiled GPU kernel time | 4,874 ms | 4,012 ms |

The 640-token continuations produced the same 64 token IDs in both modes. The 40-token continuation produced the same coherent Rayleigh-scattering text and 29/48 Hugging Face token-ID agreement in both modes. The next-token choice stayed 818, matching Hugging Face. Full prefill and continued prefill retain their earlier logits; the full-versus-eight-single-token-decode maximum absolute difference was 0.563 with cuBLAS and 0.615 with the opt-in GEMV. Repeating each decode schedule was bit-identical within its mode. The focused CUDA unit test compares GEMV output against Candle on an aligned matrix and on a 1,033-row tail case.

The revised trace replaced decode cuBLAS GEMVs with 16,192 launches of `xinfer_gemma4_bf16_gemv`; their cumulative GPU time was 2,829 ms. CUDA graph capture and the remaining model work were not optimized here. BF16 accumulation order changes, so this mode is not a bitwise replacement for cuBLAS logits. llmbda subsequently ran its 80-case tool evaluation with the flag enabled: 80/80 passed, with decode reported at 52–54 ms/token. Its three same-mode checkpoint resumptions also passed. A checkpoint written with the flag disabled diverged after resumption with the flag enabled. An adapter that persists decode state must bind this mode into its checkpoint fingerprint and reject cross-mode restore.

## Decode-only profile and further tuning

The real-checkpoint test now marks its 48-token greedy decode and optional 64-token decode after a 640-token prefix with NVTX when built with the `nvtx` feature. This separates decode from model loading and the several prefills in the whole-test profile. On an otherwise idle GB10, Nsight Systems recorded:

| Decode range | Wall per token | BF16 GEMV kernels | Paged flash decode | Other GPU kernels | Gaps between kernels |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40-token prefix, 48 steps | 50.17 ms | 41.64 ms | 0.38 ms | 3.06 ms | 5.09 ms |
| 640-token prefix, 64 steps | 51.89 ms | 41.76 ms | 1.94 ms | 3.04 ms | 5.13 ms |

The columns are an approximate decomposition of the marked wall interval. The gap column includes brief copies, memsets, and host launch delay; copies and memsets together were under 0.1 ms per token. The 20,480 × 2,560 gate/up GEMV takes 19.60 ms per token over 42 layers and reads about 4.40 GB of BF16 weights, about 225 GB/s effective. The tied 262,144 × 2,560 vocabulary projection takes another 5.73 ms and reads about 1.34 GB, about 234 GB/s effective. Those two shapes alone account for about half the token latency. Paged attention adds about 1.6 ms between these two context lengths. A graph can at most address the measured launch/gap component in this direct-model test; a larger gain requires reducing weight traffic or changing its representation, with a separate numerical and quality evaluation.

Two small GEMV changes were measured and rejected. Four instead of eight warps per block yielded 52.80/54.50 ms per token without a profiler. Loading eight BF16 values per lane through a 16-byte vector yielded 50.76/52.42 ms per token in matched Nsight ranges, compared with 50.17/51.89 ms for the existing pair-load kernel; its GEMV kernel time increased by 0.60/0.26 ms per token. The vector load changed greedy numerical choices and happened to match 48/48 Hugging Face reference tokens in this one probe, but was slower and has no broad quality validation. Both candidates were reverted. The existing pair-load kernel remains the measured faster option.

To reproduce the marked trace, build the test with `--features cuda,cutlass,nvtx`, set `XINFER_GEMMA4_BF16_GEMV=1` and `XINFER_GEMMA4_BENCH_LONG_DECODE=1`, then run it under `nsys profile --trace=cuda,nvtx`. The ranges are `gemma4_greedy_decode` and `gemma4_long_decode`.

Run the differential with `XINFER_GEMMA4_CHECKPOINT`, `XINFER_GEMMA4_REFERENCE_DIR`, and optionally `XINFER_GEMMA4_BENCH_LONG_DECODE=1`:

```bash
XINFER_GEMMA4_BF16_GEMV=1 \
XINFER_GEMMA4_CHECKPOINT=/path/to/gemma-4-E4B-it \
XINFER_GEMMA4_REFERENCE_DIR=/path/to/reference \
cargo test --release --features cuda,cutlass --test gemma4_shared_kv -- --nocapture
```
