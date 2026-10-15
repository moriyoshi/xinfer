# Nemotron-3-Nano prefill gap after the WMMA SSD and one-kernel top-k changes

On one GB10, the current Xinfer one-call path took **1,471.4 ms** for the
9,538-token Nemotron chat prompt; the saved vLLM 26.05 one-call range took
**1,020.3 ms** on the same token IDs. This is a 451.1 ms, 1.44× gap in the
matched Nsight Systems ranges. The token-ID SHA-256 is
`a32fe77947d6b586208bee6480b79bff28bd6c7a5ecf132b479985ef1195cf86`.
The checkpoint is Nemotron-3-Nano-30B-A3B NVFP4 with BF16 attention KV,
batch size one, eager resident experts, and no prefix cache. The construction
and vLLM configuration are in [the original comparison](nemotron-3-nano-gb10-prefill.md).

Xinfer was `30372978395bc4a49937e61b84629926069feca6`, built in release
mode with `cuda,cutlass,nemotron-attention-gemm,nemotron-expert-fused-quant,`
`nemotron-mamba-ssd-wmma,nvtx`. The runtime enabled the custom attention
GEMM, fused expert quantization/scatter/finalizer, WMMA SSD output with the
64×64 C×B tile, and one-kernel top-k. `XINFER_NEMOTRON_BENCH_CHUNK=9538`
gave Xinfer the same one-call boundary as vLLM. Both reported intervals
exclude model load and warmup; the Xinfer test also excludes its one decode
step. The vLLM profile is from `nvcr.io/nvidia/vllm:26.05-py3`, vLLM
`0.20.1+7124b12a.dev`, with FlashAttention 2, FlashInfer CUTLASS FP4
dense/grouped GEMMs, and Triton Mamba SSD. It was captured earlier on the
same host, not concurrently.

| Timed activity | Xinfer | vLLM | Xinfer excess |
| --- | ---: | ---: | ---: |
| Prefill range | 1,471.4 ms | 1,020.3 ms | 451.1 ms |
| Union of GPU kernels, copies, and memsets | 1,404.0 ms | 1,017.1 ms | 386.9 ms |
| All GPU kernels | 1,366.5 ms / 1,355 launches | 1,003.7 ms / 1,207 launches | 362.8 ms |
| GPU memsets | 34.3 ms / 198 | 5.4 ms / 47 | 28.9 ms |
| Time with no recorded GPU activity | 67.5 ms | 3.2 ms | 64.2 ms |

The GPU-activity union covers about **86%** of the 451 ms range difference.
The unoccupied intervals and extra zeroing matter, but removing both would
leave most of the gap. CUDA API wait durations overlap GPU work and are not
added to these totals.

| Named GPU work | Xinfer | vLLM | Difference |
| --- | ---: | ---: | ---: |
| Mamba prepare/summary/prefix/C×B/output/transposes vs. SSD/conv | 319.8 ms / 161 launches | 139.7 ms / 138 launches | 180.2 ms |
| Main attention prefill | 125.2 ms / 6 launches | 62.8 ms / 6 launches | 62.4 ms |
| Grouped NVFP4 expert GEMMs | 232.3 ms / 46 launches | 197.8 ms / 46 launches | 34.4 ms |
| Dense NVFP4 GEMMs | 81.3 ms / 80 launches | 78.4 ms / 80 launches | 2.9 ms |

These are **named kernel comparisons, not interchangeable layer totals**.
Both Mamba paths have additional pointwise and projection work. Xinfer's
WMMA output (`ssd_output`) takes 108.3 ms and its FP32 chunk summary takes
69.2 ms; vLLM's chunk output and chunk-state kernels take 40.8 and 29.3 ms.
Xinfer's C×B kernel takes 30.6 ms versus vLLM's 8.9 ms chunk BMM. The
remaining named difference includes state-prefix, transpose, decay, and
convolution preparation. Xinfer's output is already a BF16 tensor-core
kernel, but its summary still walks tokens with FP32 recurrent updates and
its output tile is slower than vLLM's Triton SSD tile. This is the largest
identifiable algorithmic gap.

Xinfer also spends 64.2 ms scattering routed expert rows, 68.8 ms
finalizing them, 62.1 ms quantizing gathered inputs, and 13.8 ms on ReLU²:
**208.9 ms** in these four named groups. vLLM fuses and arranges the same
work differently. Its 20.5 ms input expansion, 30.6 ms activation, and
36.7 ms FP4 conversion are not a one-to-one partition, so a net routed-row
penalty cannot be determined by subtracting them. It remains a substantial
Xinfer cost to target with a controlled A/B change. The grouped expert GEMM
itself is only about 34 ms slower, and dense FP4 GEMM is nearly equal;
replacing the expert GEMM alone cannot close the 451 ms gap. The one-kernel
top-k takes 3.3 ms and is no longer a dominant launch source.

The practical order is: improve the Mamba SSD summary/output and C×B tiles;
then measure fusion of routed-row quantization, activation, scatter, and
finalization; then tune the six paged attention launches. The attention
kernel's 125.2 vs. 62.8 ms is a directly measured gap, but a previous
16/256/2,048-token KV-page A/B did not materially change its time. Reduce
the remaining GPU idle intervals and zero fills after those larger kernels.

## K64 WMMA staging follow-up

The optional `XINFER_NEMOTRON_SSD_K64=1` mode stages 64 reduction values per
shared-memory load in both the 64×64 C×B kernel and the SSD output kernel.
It still performs the same ordered sequence of 16-wide WMMA operations, so
this changes synchronization and memory staging without changing the math.
It requires the `nemotron-mamba-ssd-wmma` build feature; these measurements
also enable `XINFER_NEMOTRON_SSD_WMMA=1` and
`XINFER_NEMOTRON_SSD_CB64=1`. The mode is disabled by default.

On the real NVFP4 30B checkpoint, the CUDA recurrence oracle matched the
K16 path byte for byte, including a partial chunk, zero decay, and nonzero
carried starts. The final logits and exported FP32 Mamba state also matched
byte for byte at 2,048, 9,538, and 32,768 tokens. The 32K run used four
continued 8,192-token prefill calls. These are one-pair wall measurements on
a shared host, so their small differences are indicative rather than a
service latency distribution.

| Prefill workload | K16 | K64 | Difference |
| --- | ---: | ---: | ---: |
| 2,048 synthetic tokens | 357.1 ms | 350.8 ms | 6.3 ms (1.8%) |
| 9,538 fixed chat tokens, first pair | 1,497.2 ms | 1,471.1 ms | 26.1 ms (1.7%) |
| 9,538 fixed chat tokens, reverse pair | 1,456.3 ms | 1,441.3 ms | 15.0 ms (1.0%) |
| 32,768 synthetic tokens, 8,192-token calls | 6,137.3 ms | 6,088.3 ms | 49.0 ms (0.8%) |

In separate Nsight captures of the same 9,538-token ID stream, the 46
`ssd_output` launches totaled 113.3 ms at K16 and 96.2 ms at K64; the 46
`cb_gemm64` launches totaled 31.5 and 25.6 ms. This roughly 23 ms kernel
gain is consistent with the paired wall results, but leaves most of the
451 ms matched-profile gap to vLLM. The scalar FP32 chunk summary and the
attention kernel are unchanged. K64's larger shared-memory footprint is
48 KiB per SSD output block, so other GPU architectures need separate tuning.

## Direct FP32 chunk starts follow-up

The optional `XINFER_NEMOTRON_SSD_DIRECT_STARTS=1` mode reads each carried
FP32 chunk state directly in the SSD output kernel and converts it into the
same BF16 shared-memory tile. It skips the separate state-transpose kernel
and its allocation. For a 9,538-token call this removes a 78,643,200-byte
(75 MiB) temporary buffer; for each 8,192-token continued call it removes
64 MiB. The default remains the original transpose path.

The CUDA recurrence oracle matched output bytes with nonzero carried state,
zero decay, and a partial chunk. On the real 30B checkpoint, final logits
and exported Mamba state matched the transpose path byte for byte at 2,048,
9,538, and 32,768 tokens; the 32K test used four continued calls. The
direct path also repeated its full CUDA output byte for byte.

In nearby 9,538-token Nsight captures, 46 transpose launches cost 28.1 ms.
The 46 SSD output launches rose from 98.5 to 113.1 ms with coalesced FP32
reads, leaving about 13.5 ms less combined kernel time. These sums include
the benchmark's warmup and prefill. Whole-model paired wall times were
inconclusive while shared-host load averages reached 25–47, so this is a
measured kernel and scratch reduction, not an established serving-latency
gain. The underlying FP32 chunk summary and attention costs remain.

There is substantial shared-host variation. Another capture with identical
flags took 2,274.7 ms, while its unprofiled run took 1,912.1 ms. The stable
capture above was bracketed by unprofiled 1,450.8 and 1,497.1 ms runs; the
active GB10 clock sample was 2,496 MHz. No other GPU compute process was listed before
the controlled repeats. These observations do not establish a service-latency
distribution or a causal speedup for any proposed change. Nsight Compute
hardware counters are unavailable here (`ERR_NVGPUCTRPERM`). The raw stable
Xinfer trace is `/tmp/xinfer-gap-current/full2.nsys-rep` with its SQLite
export; the vLLM trace is
`/tmp/xinfer-nemotron-prefill/nsys-vllm-9538-current.nsys-rep`.

## Cause of the remaining gap on the current fork

A subsequent capture used fork `f424ebc15e3724b266c02f2e821991120578848e`
with K64 SSD staging enabled and direct FP32 chunk starts disabled. It used
the same 9,538 token IDs, one-call boundary, BF16 KV, resident NVFP4 weights,
and GB10 as above. The Xinfer prefill range was **1,509.350 ms**, versus
**1,020.285 ms** in the saved vLLM range: a **489.065 ms** difference. The
Xinfer trace is `/tmp/xinfer-gap-current/directstarts-baseline-profile.sqlite`;
the vLLM trace is
`/tmp/xinfer-nemotron-prefill/nsys-vllm-9538-current.sqlite`. They were
captured at different times on this shared host. This section attributes the
recorded ranges; it does not assume their 38 ms difference from the earlier
Xinfer capture is a code regression.

| Activity within the marked range | Xinfer | vLLM | Difference |
| --- | ---: | ---: | ---: |
| GPU activity union | 1,442.619 ms | 1,017.068 ms | 425.551 ms |
| Kernel durations | 1,401.785 ms / 1,355 launches | 1,003.743 ms / 1,207 launches | 398.042 ms |
| Memset durations | 35.868 ms / 197 | 5.432 ms / 47 | 30.436 ms |
| Copy durations | 5.036 ms / 240 | 9.738 ms / 47 | −4.702 ms |
| No recorded GPU activity | 66.731 ms | 3.217 ms | 63.514 ms |

GPU work explains **87%** of the wall difference. Kernel, copy, and memset
durations can overlap, so their rows must not be added to reconstruct the
activity union. CUDA API durations also overlap GPU execution. Of Xinfer's
66.7 ms inactive time, 49.1 ms falls in the first 160 ms, 4.4 ms in the next
1,330 ms, and 13.3 ms at the end. The early range contains long
`cuMemAllocAsync` calls and a first `cuKernelGetFunction`/attribute setup;
the final gap follows the last GPU cast and precedes the output read. This is
mostly startup/end-of-range overhead, not a repeated idle interval between
every model layer. The differing output-read boundaries may account for part
of the final 13 ms, so that part is not a demonstrated engine-only speedup.

The largest identifiable kernel differences are:

| Work in the marked range | Xinfer | vLLM | Excess |
| --- | ---: | ---: | ---: |
| Mamba prepare, SSD summary/prefix/C×B/output/transpose/decay versus vLLM convolution and SSD | 313.37 ms | 139.66 ms | **173.71 ms** |
| Six main attention launches | 140.99 ms | 62.82 ms | **78.17 ms** |
| Grouped NVFP4 expert GEMMs | 237.56 ms | 197.84 ms | **39.72 ms** |
| Dense NVFP4 GEMMs | 78.05 ms | 78.40 ms | −0.35 ms |
| BF16 CUTLASS GEMMs | 64.64 ms | 62.84 ms | 1.80 ms |

These are corresponding **named kernels**, not full-layer totals. The first
three rows account for about 292 ms of the 398 ms kernel-duration difference;
about 106 ms remains spread across pointwise, routing, quantization, and
other kernels. Xinfer's routed-row `finalize`, scatter, gathered-input
quantization, and ReLU² alone take 219.5 ms. vLLM partitions equivalent work
among input expansion, activation, FP4 conversion, and many generic
elementwise kernels; subtracting just their named times would misattribute
the residual. A controlled routed-row A/B is still needed to isolate that
opportunity.

The Mamba difference has a clear implementation basis. Xinfer's
`nemotron_mamba_chunk_summary` in `src/models/nemotron_h/prefill/scan.cu`
walks the 128 tokens of each chunk in an FP32
state-update loop (71.7 ms), while vLLM's Triton SSD forms chunk state using
tiled `tl.dot` operations. Xinfer's 16-wide WMMA C×B and output kernels,
including separate state transpose, take 149.3 ms; the corresponding vLLM
chunk BMM and scan take 49.6 ms. The distinct recurrence order and precision
mean a replacement needs a numerical and deterministic-continuation check,
not just a timing check. Xinfer's current paged attention kernel in
`src/models/nemotron_h/attention_gemm/flash_prefill_paged.cuh` uses a
64-query × 32-key tile with eight warps; the recorded vLLM FlashAttention 2
kernel has a 128-query × 64-key trait with four warps. Both traces launch
4,800 attention CTAs. Tile shape, memory traffic, and scheduling therefore
deserve investigation; the CTA count alone does not explain the 78 ms gap.

There is also **30.4 ms of excess GPU memset work**. Xinfer zeros 3.342 GB
in 17 tensors of `9538 × 10304 × BF16`, 1.629 GB in 23 tensors of
`9538 × 3712 × BF16`, and 2.051 GB in 40 tensors of
`9538 × 2688 × BF16`. The pinned `attention-rs` NVFP4 linear path allocates
the output with `Tensor::zeros((m, n), ...)` before the hardware CUTLASS
GEMM. Its SM121 epilogue declares `ElementC = void`, so successful hardware
GEMMs do not use the previous output values. These large clears are a
specific optimization candidate, subject to proving full output coverage,
error propagation, and parity for all supported paths and shapes. vLLM's
large zero fills total 1.179 GB and 5.420 ms in this range.

The measured priority is therefore: (1) Mamba chunk summary and output
tiling, about 174 ms of named excess; (2) paged attention, about 78 ms;
(3) quantify and fuse remaining routed-row work; (4) remove proven
unnecessary hardware-FP4 output clears, at most roughly 30 ms of recorded
memsets; (5) reduce first-use allocation/kernel-setup latency. The grouped
expert GEMM contributes about 40 ms, and dense FP4/BF16 GEMMs are close to
vLLM, so a GEMM rewrite alone cannot close the measured gap.
