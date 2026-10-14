# Nemotron-H routed expert quantization on GB10

The Nemotron-3-Nano-30B-A3B NVFP4 path already uses a grouped CUTLASS FP4
tensor-core GEMM. Before that GEMM, the pinned `attention-rs` implementation
gathers every routed BF16 row into a second buffer, then quantizes that buffer
to packed E2M1 activations and swizzled E4M3 scales. Xinfer's opt-in fused
kernel reads the routed source row directly while quantizing. It preserves the
grouped GEMM, packed checkpoint weights, routing order, scale format, and
scatter. No model weights are requantized.

Build with `--features cuda,cutlass,nemotron-expert-fused-quant` and set
`XINFER_NEMOTRON_EXPERT_FUSED_QUANT=1` at process startup. Both switches are
required. The default path is unchanged. The optimized path requires GB10
`sm_121`, BF16 inputs, eager multi-token prefill, zero-offset contiguous input
and route tensors, and calibrated input scales. Decode, other GPUs and dtypes,
and `XINFER_NVFP4_ONLINE_SCALE` use the existing `attention-rs` path. Building
the feature requires a CUDA toolkit that accepts `sm_121f`.

## Real-checkpoint checks

Measured 2026-10-14 on one GB10 with the five-shard local
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` checkpoint. All runs used one
sequence, 8,192-token prefill chunks, eager experts, BF16 attention KV, release
`cuda,cutlass,nemotron-expert-fused-quant`, and a 16 GiB MemAvailable watchdog.
The 9,538-token prompt is the exact token stream from
`docs/nemotron-3-nano-gb10-prefill.md`; 2,048 and 32,768 tokens use the
existing repeated-text test stream. Times are one warm checkpoint-load run per
variant, not medians.

| Prefix | Original prefill | Fused prefill | Change | Logits |
| ---: | ---: | ---: | ---: | --- |
| 2,048 | 454.5 ms | 422.7 ms | -7.0% | Prefix and 64 continuation steps byte-identical |
| 9,538 | 2,142.2 ms | 1,916.2 ms | -10.6% | Prefix and next step byte-identical |
| 32,768 | 8,209.2 ms | 7,606.7 ms | -7.3% | Prefix and next step byte-identical |

The first 2,048-token baseline model load was much colder than the fused load;
the 9,538 and 32,768-token pairs had similar model-load times. A focused CUDA
test also compares the packed E2M1 bytes and swizzled E4M3 scale bytes against
the original gather-then-quant path at both expert widths (1,856 and 2,688).

In matched 9,538-token Nsight captures, the named gather kernels fell from
174.3 ms across 92 launches to zero. The fused quant kernels took 58.6 ms,
versus 65.2 ms for original quantization. Grouped GEMM stayed at 311.9 vs
311.5 ms and scatter at 164.2 vs 163.7 ms. All GPU kernel time fell from
1,935.7 to 1,755.0 ms and kernel launches from 4,067 to 3,975. These are
profiled kernel sums; the unprofiled prefill times above are the serving-side
comparison. The remaining grouped GEMM and scatter costs still limit parity
with the previously measured vLLM first-token time near 1,036 ms at 9,538
tokens.

The isolated tile experiment used 128 experts, top-6, 57,228 routed rows and
the checkpoint's up/down dimensions. The pinned 128×128×128 grouped GEMM took
about 4.4/5.7 ms for up/down. A 128×256×128 tile took about 32/36 ms, a
128×128×256 tile was effectively unchanged, and CUTLASS automatic scheduling
at 128×128×128 was also unchanged. Those synthetic times did not motivate a
GEMM tile change.
