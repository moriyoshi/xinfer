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

## Why the one-call prefill still trails vLLM

The one-call comparison above removes the 8,192-token Xinfer call boundary.
It leaves 1,613.7 ms for Xinfer versus 1,020.3 ms for vLLM on the same
9,538-token prompt. The Xinfer trace has 1,976 kernels totaling 1,537.5 ms;
the vLLM `BENCH_PREFILL_9538` range has 1,207 kernels totaling 1,003.7 ms.
Both runs are therefore predominantly GPU-bound. Including copies and memsets,
the GPU activity union spans 1,575.4 of Xinfer's 1,617.7 ms captured window
and 1,017.1 of vLLM's 1,020.3 ms range. Xinfer's 569 GPU memsets take
37.3 ms; vLLM's 47 take 5.4 ms. Eliminating host gaps and zeroing alone
cannot close the roughly 0.59-second one-call wall gap.

| Named kernel work, one 9,538-token call | Xinfer | vLLM | Observation |
| --- | ---: | ---: | --- |
| Mamba prepare, chunk summary/prefix/scan vs. SSD/conv | 437.0 ms | 139.7 ms | Largest named difference; **not** complete layer time on either side |
| Main attention prefill | 123.9 ms | 62.8 ms | Six launches on each side |
| Grouped NVFP4 expert GEMMs | 228.4 ms | 197.8 ms | Same 46 up/down launches; 30.6 ms difference |
| Dense NVFP4 GEMMs | 74.7 ms | 78.4 ms | Xinfer is already comparable |
| BF16/FP32 CUTLASS GEMMs | 60.8 ms | 62.8 ms | Xinfer is already comparable |

The Mamba difference follows from the algorithms in the exact benchmarked
sources. Xinfer's [`prefill/scan.cu`](../src/models/nemotron_h/prefill/scan.cu)
summarizes each 256-token chunk by a scalar recurrent pass from zero, passes
its final state to the next chunk, and then performs a second scalar recurrent
pass to produce every output. Its 23 `nemotron_mamba_chunk_scan` launches take
313.8 ms (13.6 ms per layer); summary adds 63.9 ms, prefix 20.9 ms, and
prepare 38.4 ms. vLLM's `mamba_chunk_scan_combined_varlen` uses a chunk-state
kernel, inter-chunk state passing, a C×B matrix product, and an intra-chunk
scan that consumes that product. Its named SSD/conv kernels total 139.7 ms.
Each implementation also has pointwise and projection work outside these
names, so 297.3 ms is a comparison of named kernels, not a measured net
benefit from replacing Xinfer's Mamba path. A chunked SSD matrix path is
the leading candidate; the previously tried scalar output-correction kernel
was slower and was discarded.

The attention kernels launch the same 4,800 thread blocks per layer. Xinfer's
[`flash_prefill_paged.cuh`](../src/models/nemotron_h/attention_gemm/flash_prefill_paged.cuh)
uses a 64-query by 32-KV tile, 8 warps, and explicit per-row paged loads into
shared memory. vLLM uses a FlashAttention-2 kernel with a 128×64×128 template
and 4 warps. The larger KV tile and different load/softmax pipeline are
plausible reasons for the 61.1 ms difference; the trace alone cannot assign
the gain to either one. vLLM logs 2,096-token attention pages, whereas the
Xinfer benchmark used 16-token pages. To isolate page size, the real-checkpoint
benchmark now accepts `XINFER_NEMOTRON_BENCH_KV_BLOCK` (power of two). With
the same one-call 9,538-token prompt and optimized flags, unprofiled prefill
was 1,681.6 and 1,687.5 ms with 16-token pages (bracketing controls),
1,678.8 ms with 256-token pages, and 1,664.7 ms with 2,048-token pages.
All prefix and next-step logit bytes matched. The separate 256-page Nsight
capture recorded 127.7 ms of attention versus 123.9 ms in the earlier
16-page capture; this small reversed difference is within run variation.
Larger pages alone did not materially improve this attention implementation.

The custom grouped expert GEMM hypothesis has limited remaining scope: both
engines already use packed FP4, block-scaled sm_120 CUTLASS grouped tensor-core
GEMMs. Their difference is about 31 ms of the 594 ms wall gap. Xinfer still
spends 63.6 ms on expert up-scatter, 68.8 ms on finalization, 62.4 ms on fused
activation quantization, and 57.0 ms on ReLU²; vLLM fuses some corresponding
work differently, so these are not direct savings estimates. Xinfer also has
75.6 ms of BF16→FP32 casts, 47.5 ms of FP32→BF16 casts, and 53.3 ms of FP32
RMSNorm kernels. vLLM has its own pointwise operations. Reducing redundant
format changes and routed-row traffic is the next area to measure after SSD
and attention, with exact output controls.

Nsight Compute could not collect hardware counters on this host:
`ERR_NVGPUCTRPERM`. The analysis is based on Nsight Systems timing, launch
geometry, source inspection, and the page-size A/B checks. The page-size
results are single runs on a shared host, not a serving latency distribution.
The 2,048-page trace is not available, and the vLLM attention page size is
2,096 rather than exactly 2,048. Decode was not profiled in this follow-up.
The page-size run artifacts are `/tmp/xinfer-kvblock{16,256,2048}-logits.bin`
and `/tmp/xinfer-kvblock256-profile.nsys-rep` on this host.

## Follow-up: direct current-chunk K/V reads

The Nemotron custom attention prefill now reads the current call's contiguous
post-transform K/V tensors directly. It still writes those tensors to the
paged cache for later calls, and reads earlier prefix positions from that
cache during continued prefill. The direct path requires matching host context
and cumulative query lengths; otherwise it uses the original paged lookup.
`XINFER_NEMOTRON_DIRECT_PREFILL_KV=0` disables the direct reads for an A/B
control. The specialization itself still requires
`XINFER_NEMOTRON_ATTENTION_GEMM=1` and its existing shape/length gates.

On the 30B NVFP4 checkpoint, the same 9,538-token prompt in one prefill call
measured 1,700.3 and 1,726.2 ms with direct K/V around a 1,773.1 ms paged
control; a later default-on smoke measured 1,755.5 ms. With 32,768 tokens in
four 8,192-token calls, direct K/V measured
7,200.6 ms and paged K/V 7,294.7 ms. The complete prefix plus next-token
logit buffers were byte-identical within each comparison. These are small
single-run wins on a shared host, not stable serving latency estimates.
The 2,048-token real benchmark does not enter this custom attention path
because its existing gate requires `max_context_len >= 4096`.

The expert GEMM path also skips six caller-side metadata zero fills per GEMM.
The expert-count kernel initializes its own count buffer, the offset kernel
writes every offset, and metadata construction writes every remaining word.
Real-checkpoint logits remained byte-identical. The 9,538-token prefill was
1,639.7 ms after this change versus a nearby 1,633.8 ms control, so a wall
gain from zero-fill removal was not established. The 2,048-token checkpoint
smoke measured 389.2 ms prefill and 42.9 ms for one decode step; this length
does not enter the custom attention specialization.

Other trials did not justify changing the default path. Mamba scan chunks of
64, 128, 256, and 512 tokens measured 1,716.6, 1,670.7, 1,642.9, and
1,827.9 ms respectively on the same 9,538-token case. A ten-warp scan took
1,648.2 ms versus 1,633.8 ms for eight warps. A CUBLAS-based SSD scan passed
focused state/continuation parity but took 3,686 ms versus about 1,614 ms on
the full checkpoint, so it was discarded. The pinned FlashAttention varlen
wrapper returned incorrect BF16 context on SM121 (maximum absolute error
about 4.8 in a small CUDA fixture). FlashInfer ragged reduced isolated
9,538-row attention from about 29 to 13 ms per layer, but the real checkpoint
measured 1,780.5 ms with it versus 1,778.8 ms in the same feature-enabled
binary without it; its prefix/next-token logits differed by up to 3.02, with
the same top choice at both measured positions. That trial was discarded.

This change narrows the attention portion modestly and does not establish
parity with vLLM. The prior matched Nsight result still points to Mamba's
two scalar state passes (437 ms here versus 140 ms of named SSD/conv kernels
in vLLM) as the largest algorithmic gap. A fused tensor-core SSD path with
bounded intermediate storage remains the next substantial prefill task.

## vLLM Mamba kernel and GB10 binary inspection

The profiled NVIDIA vLLM 26.05 image contains vLLM
`0.20.1+7124b12a.dev`. Its `ssd_combined.py` calls five Triton stages:
chunk cumulative decay, chunk state, state passing, C×B chunk product, and
chunk output scan. The checkpoint sets `chunk_size=128`, 64 Mamba heads,
64 channels per head, eight B/C groups, and 128 state channels. For each
128-token chunk, the state and output stages use `tl.dot` matrix products;
only the much smaller inter-chunk state passing is sequential. In contrast,
Xinfer's `nemotron_mamba_chunk_summary` and `nemotron_mamba_chunk_scan` both
iterate through every token in each 256-token chunk with scalar FP32 state
updates. Changing Xinfer's chunk size alone did not help, as measured above.

The vLLM image's Triton sources are under
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/`.
We compiled those sources on the GB10 with synthetic tensors matching the
checkpoint's Mamba dimensions, BF16 X/B/C, FP32 delta and carried state,
and two 128-token chunks. Triton's autotune cache selected the following
tiles in this small synthetic run; these are **not** a measurement of the
9,538-token production specialization:

| Stage | Selected tile | Warps | Pipeline stages |
| --- | --- | ---: | ---: |
| Chunk state, X×scaled B | 64×128×64 | 4 | 2 |
| C×B | 128×256×64 | 8 | 3 |
| Chunk output scan | 64×64×64 | 4 | 1 |

`nvdisasm` of a compiled chunk-output-scan cubin showed
`HMMA.16816.F32.BF16` instructions. Compiled chunk-state and C×B variants
also contained BF16 HMMA instructions. This establishes that vLLM's matrix
work reaches the GB10 tensor cores; it does not establish achieved occupancy
or bandwidth because Nsight Compute counters are unavailable on this host.
The scalar FP32 state updates in Xinfer's current CUDA scan cannot use those
matrix instructions. The earlier Xinfer SSD attempt used FP32 CUBLAS GEMMs,
large FP32 intermediates, and extra conversions, so its slowdown does not
test this BF16 tensor-core design.

At 9,538 tokens, a 128-token vLLM layout has 75 chunks. A full FP32 chunk
state tensor would be about 157 MB and a full FP32 C×B tensor about 39 MB
per Mamba layer. A Xinfer implementation should reuse bounded per-layer
scratch and avoid host synchronization, preserving FP32 carried state and
the existing state envelope. The first useful gate is an isolated 128-token
SSD layer benchmark with nonzero carried state, partial final chunks, and
BF16 output parity measurements. Only then should it be tested on the 30B
checkpoint for deterministic repeated prefill and decode, continuation
logits, and 2K/32K end-to-end latency. The source and synthetic cubins used
for this inspection are under `/tmp/xinfer-vllm-mamba-source/` on this host.
