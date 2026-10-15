# Nemotron-H fused expert ReLU² and NVFP4 quantization

With `--features cuda,cutlass,nemotron-expert-fused-quant`, set
`XINFER_NEMOTRON_EXPERT_FUSED_QUANT=1` and
`XINFER_NEMOTRON_EXPERT_FUSED_RELU_QUANT=1` to enable this opt-in prefill path.
It reads the routed up-projection result, applies ReLU² with the same BF16
rounding as the separate activation, and quantizes it for the grouped down
GEMM in one kernel. It avoids writing and rereading the intermediate ReLU²
tensor. Decode, unsupported shapes, and other dtypes retain the existing path.
The existing `XINFER_NEMOTRON_EXPERT_FUSED_FINALIZE=1` setting remains
independent.

## GB10 real-checkpoint result

The 30B NVFP4 checkpoint at `xinfer-artifacts/nemotron-3-nano` ran on one GB10
with eager experts and BF16 KV. The release binary used
`cuda,cutlass,nemotron-attention-gemm,nemotron-expert-fused-quant,`
`nemotron-mamba-ssd-wmma`, with the Mamba WMMA and existing fused expert
quant/scatter/finalizer switches enabled. A guard required at least 64 GiB
host `MemAvailable` and no other GPU compute process at each start. The 32K
prompt was four continued 8,192-token prefill calls; all times exclude model
load. The comparison uses the final compile-time specialized kernel variants,
which leave the inactive path free of a per-value branch. These are individual
runs on a shared host, not a serving latency distribution.

| Prefix | Separate ReLU² | Fused ReLU² + quant | Change |
| --- | ---: | ---: | ---: |
| 2,048, pair 1 | 373.821 ms | 357.033 ms | 4.49% faster |
| 2,048, reverse pair | 373.565 ms | 359.222 ms | 3.84% faster |
| 8,192, pair 1 | 1,397.234 ms | 1,345.705 ms | 3.69% faster |
| 8,192, reverse pair | 1,377.216 ms | 1,331.564 ms | 3.31% faster |
| 8,192, Nsight pair | 1,384.694 ms | 1,344.946 ms | 2.87% faster |
| 32,768, pair 1 | 6,821.558 ms | 6,865.113 ms | 0.64% slower |
| 32,768, reverse pair | 6,925.775 ms | 6,831.937 ms | 1.35% faster |

Prefill logits and exported Mamba state were byte-identical for every matched
prefix. The focused CUDA test compares the generated FP4 bytes and swizzled
scale bytes against separate ReLU², gather, and quantization at both 1,856
and 2,688 expert widths. The test passes. In the 8K Nsight pair, the
`nemotron_relu_squared` kernels fell from 138 launches / 52.85 ms to 92
launches / 12.68 ms. The two quantizer variants together took 57.43 ms,
versus 56.69 ms for the original quantizer, a net savings of about 39.4 ms
of named kernels across the benchmark's warmup and timed prefill. The
corresponding timed prefill wall-time difference was 39.75 ms; the two totals
are not a direct phase-matched comparison.

The 32K pairs straddle zero; no 32K wall-time win is established. The fused
path is disabled by default because the measured 8K gain is modest and
currently verified only on the GB10 30B checkpoint. It does not alter the
grouped CUTLASS GEMM or close the remaining full-model gap against vLLM.
