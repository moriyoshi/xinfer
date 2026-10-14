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

## Fresh vLLM profile against fused Xinfer

On the same GB10, a fresh Nsight Systems 2025.3.2 capture of the warmed
NVIDIA `nvcr.io/nvidia/vllm:26.05-py3` container (vLLM
`0.20.1+7124b12a.dev`) took **1,020.3 ms** for the same 9,538 token IDs.
The container and BF16-KV metadata overlay are described in
`nemotron-3-nano-gb10-prefill.md`. It used eager execution, batch size one,
all experts resident, `gpu_memory_utilization=0.35`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0` so Nsight could capture its worker.
The `BENCH_PREFILL_9538` range excludes model loading, warmup, and FP4
autotuning. Minimum observed host `MemAvailable` was 56.0 GiB. The same
Xinfer revision `db829fb890da048806393340b000b72d759b3dec` with fused
quantization measured 1,900–1,916 ms in its
unprofiled runs, a roughly **1.86–1.88×** vLLM gap.

| Named GPU work in the 9,538-token prefill | Xinfer fused quant | vLLM |
| --- | ---: | ---: |
| Mamba scan/summary/prepare/prefix vs. SSD scan/state/conv/BMM | 452.5 ms / 184 launches | 139.7 ms / 138 launches |
| Grouped NVFP4 expert GEMMs | 311.5 ms / 92 launches | 197.8 ms / 46 launches |
| Main attention prefill | 172.2 ms / 12 launches | 62.8 ms / 6 launches |
| Xinfer expert scatter vs. vLLM expert input expansion | 163.6 ms / 92 launches | 20.5 ms / 23 launches |
| **All GPU kernels, including unlisted work** | **1,755.0 ms / 3,975 launches** | **1,003.7 ms / 1,207 launches** |

These named rows identify different implementations and **are not an additive
or like-for-like layer-time partition**. Xinfer's runner processes this prompt
as 8,192 plus 1,346 tokens, so each hybrid layer runs twice; vLLM processes
all 9,538 tokens in one prefill chunk. This accounts for the 92 vs. 46 expert
GEMM launches and 12 vs. 6 attention launches, and makes the per-kernel
differences unsuitable as a pure GEMM efficiency ratio. The total times still
describe the actual configured paths. vLLM's 1,003.7 ms of kernels occupy
98.4% of its measured 1,020.3 ms range, so its host schedule is not a large
unexplained portion of this profile. Its 3 CUDA event synchronizations spent
820 ms waiting for GPU work; that API time overlaps kernels and is not an
additional cost.

The grouped expert GEMM difference is about 114 ms in this run. Even fully
eliminating that difference would close only a small part of the roughly
880–896 ms wall gap. The Mamba core kernels differ by about 313 ms, attention
by 109 ms, and Xinfer's expert scatter alone costs 164 ms. The Xinfer trace
also contains 58.6 ms of fused expert quantization, 56.0 ms of ReLU², and
41.1 ms of route-combine kernels; vLLM fuses some corresponding work into
other kernels, so subtracting individual rows cannot predict an optimization's
gain. A custom expert GEMM remains a possible target, but matching vLLM
requires reducing Mamba, attention, and expert row-movement work as well.

Raw trace: `/tmp/xinfer-nemotron-prefill/nsys-vllm-9538-current.nsys-rep`
and its SQLite export; current Xinfer trace:
`/tmp/xinfer-expert-tune/fused.sqlite`. These are local artifacts, not files
tracked by the repository.
