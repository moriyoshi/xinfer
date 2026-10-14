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

## Expert output finalization follow-up

The opt-in `XINFER_NEMOTRON_EXPERT_VECTOR_SCATTER=1` path converts four
FP32 grouped-GEMM values to BF16 per thread and skips the zero fill of the
fully overwritten routed output. With
`XINFER_NEMOTRON_EXPERT_FUSED_FINALIZE=1`, the down projection instead
inverts the native expert-order permutation on the GPU and directly combines
the six BF16-rounded expert values, FP32 route weights, and BF16 shared
expert into the final BF16 output. Both switches require the build feature
`nemotron-expert-fused-quant` and its existing runtime switch
`XINFER_NEMOTRON_EXPERT_FUSED_QUANT=1`. The default, decode, unsupported
dtype/device, and non-grouped paths are unchanged. Packed NVFP4 checkpoint
weights and scales are unchanged. The finalizer is deliberately limited to
BF16, sm_121, eager multi-token prefill with complete native routing.
`XINFER_NEMOTRON_FUSED_MOE_COMBINE=0` also keeps the original combination
path for comparisons.

On the real 30B checkpoint, the new path matched the old path's logit **bytes**
for 2,048 prefix + 64 teacher-forced positions, the exact 9,538-token chat
prefix + one step, and 32,768 prefix + one step. A focused CUDA test matches
the vector scatter to the original scatter at both expert widths and matches
the finalizer to original scatter-plus-combine with a permuted six-route
fixture. Both tests compare BF16 bits, including the rounding boundary.

| Real-checkpoint prefill | Original fused-quant path | Vector scatter + finalizer |
| --- | ---: | ---: |
| 2,048 tokens | 423.0 ms | 390.7 ms |
| 9,538 tokens, three runs | 2,342.1 / 1,832.2 / 1,867.4 ms | 1,702.0 / 1,704.7 / 1,700.8 ms |
| 32,768 tokens | 7,023.8 ms | 6,523.7 ms |

The first 9,538-token baseline is an outlier; the other two baseline runs
are 1,832–1,867 ms. The three optimized runs cluster near 1,702 ms.
These are separate warm model-load processes on a shared host, so they do
not establish a serving throughput distribution. Their 9,538-token median
gap against the fresh vLLM profile above falls from 1.83× to 1.67×; the
new path does **not** reach parity.

Nsight captures of the same 9,538-token input give a more focused comparison:

| Profiled work | Original | Vector + finalizer |
| --- | ---: | ---: |
| Expert scatter and route-combine kernels | 165.0 + 40.0 ms | 64.0 ms up-scatter + 69.5 ms finalizer + 0.2 ms route inversion |
| All GPU kernels | 1,703.7 ms / 3,975 launches | 1,634.9 ms / 3,975 launches |
| GPU memsets | 100.3 ms / 20.69 GB | 36.2 ms / 7.55 GB |

The kernel count stays constant because each removed down-scatter and combine
pair is replaced by route inversion and finalization. Kernel and memset times
exclude host-side work and cannot be added to the unprofiled prefill times.
The trace and guarded run outputs are in `/tmp/xinfer-vector-scatter/` on the
measurement host. A 16-warp Mamba scan experiment measured 1,712 ms on the
same input versus roughly 1,701–1,705 ms with eight warps; it was reverted.
Mamba scan, grouped GEMM, and attention remain the largest named kernel gaps
against vLLM.

The real-checkpoint benchmark now accepts `XINFER_NEMOTRON_BENCH_CHUNK` for a
test-only call-boundary control. With one 9,538-token Xinfer call, original
fused quantization took 1,743.9 ms and the vector-scatter/finalizer path took
1,613.7 ms; their prefix and next-step logit bytes matched. This more closely
matches vLLM's one-call schedule and leaves a **1.58×** wall gap against its
1,020.3 ms profile. Changing the Xinfer call boundary from 8,192+1,346 to
9,538 changed logits (maximum absolute difference 3.3125 for prefix logits
and 2.875 for next-step logits; both top choices stayed the same). It is a
numerical schedule change, so the one-call result is a diagnostic, not a
transparent replacement for existing callers.

The one-call optimized trace has 1,537.5 ms of GPU kernels. Its named Mamba
prepare/summary/prefix/scan kernels total 437.0 ms, versus 139.7 ms for the
named vLLM SSD/conv kernels. Grouped expert GEMM is 228.4 vs 197.8 ms,
and main attention is 123.9 vs 62.8 ms. This makes the remaining recurrent
path the largest measured opportunity. The Xinfer one-call trace is
`/tmp/xinfer-vector-scatter/finalize-onecall-9538.nsys-rep`.
