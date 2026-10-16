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

The `XINFER_NEMOTRON_SSD_DIRECT_STARTS=1` mode reads each carried
FP32 chunk state directly in the SSD output kernel and converts it into the
same BF16 shared-memory tile. It skips the separate state-transpose kernel
and its allocation. For a 9,538-token call this removes a 78,643,200-byte
(75 MiB) temporary buffer; for each 8,192-token continued call it removes
64 MiB. This mode is now the default when the opt-in SM121 SSD WMMA path is
enabled; set `XINFER_NEMOTRON_SSD_DIRECT_STARTS=0` for the original transpose
path.

The CUDA recurrence oracle matched output bytes with nonzero carried state,
zero decay, and a partial chunk. On the real 30B checkpoint, final logits
and exported Mamba state matched the transpose path byte for byte at 2,048,
9,538, and 32,768 tokens; the 32K test used four continued calls. The
direct path also repeated its full CUDA output byte for byte.

A subsequent 9,538-token, 16-step real-checkpoint comparison after the
attention warp-local change again matched all 8,912,896 logit bytes and all
49,930,472 snapshot bytes. The unprofiled prefill calls measured 2,545.3 ms
with transpose and 1,466.2 ms with direct starts, but model load varied from
52.5 to 80.3 seconds on this shared host. Those two wall measurements do not
establish a 1.08-second kernel gain; the paired Nsight kernel result below is
the narrower evidence for the change.

In nearby 9,538-token Nsight captures, 46 transpose launches cost 28.1 ms.
The 46 SSD output launches rose from 98.5 to 113.1 ms with coalesced FP32
reads, leaving about 13.5 ms less combined kernel time. These sums include
the benchmark's warmup and prefill. Whole-model paired wall times were
inconclusive while shared-host load averages reached 25–47, so this is a
measured kernel and scratch reduction, not an established serving-latency
gain. The underlying FP32 chunk summary and attention costs remain.

## SSD output shared scratch and 128-row tiles

The SM121 SSD WMMA output kernel now reuses its shared-memory input-tile
buffer for the FP32 accumulators after the final MMA. Each thread saves its
main accumulator values in registers before the state accumulator overwrites
that shared region. This reduces K64 staging from 48 KiB to 16 KiB for a
64-row block and lets a 128-row block fit in 32 KiB. The per-output WMMA
fragments and reduction order are unchanged.

On the same 9,538-token real 30B prompt, one CUDA trace per setting measured
the following sums over the 23 prefill SSD output launches (excluding the 23
short warmup launches):

| Output tile | Shared scratch | SSD output GPU time |
| --- | ---: | ---: |
| 64 rows, original buffers | 48 KiB | 106.480 ms |
| 64 rows, reused buffer | 16 KiB | 104.955 ms |
| 128 rows, reused buffer | 32 KiB | 93.518 ms |

The 64- and 128-row reused-buffer measurements came from the same binary;
their chunk-summary and chunk-prefix sums differed by under 0.2 ms. The
128-row output is the default only when the opt-in SM121 SSD WMMA path is
enabled. Set `XINFER_NEMOTRON_SSD_OUTPUT_ROWS=64` to restore the old row
geometry. A 32-row trial cost 168.980 ms and was removed.

The 128-row run matched all 8,912,896 saved real-checkpoint logit bytes and
all 49,930,472 Mamba snapshot bytes against the prior 64-row path; a second
prefill and 16 decode steps repeated bit for bit. The unprofiled and
profiled whole-model wall times varied enough to obscure this roughly
11 ms kernel gain, so it is not a measured serving-latency improvement.
Trace exports: `/tmp/nemotron-ssd-output64-shared.sqlite` and
`/tmp/nemotron-ssd-output128-shared.sqlite`.

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

### Optional 128-wide SSD output reduction

With `XINFER_NEMOTRON_SSD_K64=1`, setting
`XINFER_NEMOTRON_SSD_OUTPUT_K128=1` stages all 128 reduction values per
SSD output block. It retains the ordered 16-wide WMMA operations but uses
48 KiB of shared scratch rather than K64's 32 KiB. The mode applies only to
the opt-in SM121 WMMA path; the default remains K64. The launcher rejects a
K128 request without K64 enabled.

The focused CUDA recurrence test matched K64 output bytes with nonzero
carried state, zero decay, and a partial chunk. On the real 30B checkpoint,
the 9,538-token prefill and 16 decode steps matched the published K64 path
byte for byte in all 8,912,896 saved logit bytes and 49,930,472 exported
Mamba-state bytes. Two same-binary Nsight pairs used opposite orders:

| Order | K64 SSD output, 23 launches | K128 SSD output, 23 launches |
| --- | ---: | ---: |
| K64 then K128 | 92.229 ms | 91.121 ms |
| K128 then K64 | 92.519 ms | 91.624 ms |

The named-kernel difference is 0.895–1.109 ms, or about 1% of SSD output
time and under 0.1% of the 9,538-token prefill. The four corresponding
whole-prefill wall times were 1,434.4, 1,433.0, 1,423.8, and 1,450.9 ms in
run order; they do not establish a reliable end-to-end speedup. K128 was
therefore left opt-in. The same-binary traces are
`/tmp/nemotron-ssd-output-k{0,1}-{samebinary,reverse}.sqlite`.

### Independent prefix warp geometry

`XINFER_NEMOTRON_PREFIX_WARPS=2` now selects two warps only for the FP32
chunk-boundary prefix kernel. The summary remains at its faster four-warp
geometry, and the ordinary default remains four warps for both. The module
cache key includes both geometry choices. No recurrence formula, chunk
boundary, or exported state layout changes.

An initial two-warp trial applied to **both** summary and prefix matched
real-checkpoint bytes but made the 9,538-token summary 13.6 ms slower while
reducing prefix time by 4.3 ms. It was discarded. In the same-binary
prefix-only comparison, the 23 measured prefill prefix launches took
41.845 ms at four warps and 38.555 ms at two; summary time was 70.909 and
70.805 ms respectively. A single real-checkpoint 32,768-token pair, split
into four continued 8,192-token calls, took 5,980.9 and 5,943.7 ms in
prefill wall time. The corresponding 9,538-token wall times were 1,471.2
and 1,470.0 ms; shared-host variability prevents attributing either wall
difference entirely to the prefix change.

All 9,538-token logits and exported Mamba-state bytes matched between
geometries; the 32,768-token continued-prefill logits and state did too.
The two-warp mode repeated prefill, 16 decode logits, and state bit for bit.
Its compiled SM121 prefix kernel used 168 registers per thread with no
reported spills. This tuning was measured on GB10 only, so it remains opt-in.
Trace exports are `/tmp/nemotron-prefix-{4,2}warp.sqlite`.

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

## Further attribution on the current epic branch (2026-10-16)

A fresh Nsight capture on `648cb0141dd777e1f4978a4649691093fae0852e`
used the same 9,538 token IDs and the flags above, including the later K64
SSD and warp-local attention modes. Its `BENCH_PREFILL_NEMOTRON` range was
1,476.752 ms. The saved vLLM `BENCH_PREFILL_9538` range was 1,020.285 ms.
These are separate captures on a shared host; their 456.467 ms difference is
an attribution baseline, not a paired service-latency estimate. Raw SQLite
exports are `/tmp/nemotron-current-epic-gap.sqlite` and
`/tmp/xinfer-nemotron-prefill/nsys-vllm-9538-current.sqlite`.

The largest newly identified repeated path is the residual plus layer RMSNorm.
Nemotron-H calls `NormX::forward` before each of its 52 mixers and adds the
residual afterward. Because its norm weights and arithmetic are FP32, the
BF16 hidden state is cast to FP32, normalized, and cast back to BF16. In the
captured Xinfer range the 52 BF16 residual-add kernels took 41.46 ms, the
53 BF16-to-FP32 casts immediately before layer/final norms took 44.36 ms,
the 53 FP32 RMSNorms took 53.05 ms, and the 53 casts immediately after them
took 40.70 ms: about **179.6 ms** of named GPU work. The saved vLLM range
contains 52 `fused_add_rms_norm` kernels totaling **45.78 ms**. The roughly
**134 ms difference** is a strong optimization target, but it is not a
measured speedup: a Xinfer fusion must preserve the exact residual placement,
BF16 rounding, and continued-prefill behavior, then quantify logits and
snapshot state against the current path.

The routed-expert path has another concrete data-movement opportunity. After
the grouped up GEMM writes expert-sorted FP32 rows, Xinfer scatters and rounds
them into BF16 route-assignment order. Its down-projection quantizer then
gathers those same assignments back into expert-sorted order. With the
existing fused-ReLU-quant flag enabled, the scatter alone took **65.23 ms**
over 23 layers; gathered quantization took **62.07 ms** over 46 launches.
Keeping the intermediate in sorted order while preserving the FP32-to-BF16
rounding boundary could remove the scatter and part of the gather, but needs
a byte-parity test of down-projection packed inputs and scales. The existing
`XINFER_NEMOTRON_EXPERT_FUSED_RELU_QUANT=1` already avoids a separate
activation pass: in a nearby single capture, the marked range dropped from
1,476.752 to 1,430.624 ms and ReLU² kernels from 58.35 to 15.12 ms.
That one-pair whole-range difference is indicative; the activation-kernel
change is the narrower evidence. The flag requires the
`nemotron-expert-fused-quant` build feature and the fused expert quant path.

The 46 grouped CUTLASS expert GEMMs took 236.64 ms with the fused-ReLU flag,
versus 197.84 ms in the saved vLLM range. Xinfer uses an SM120 block-scaled
pointer-array ping-pong collective with a 128×128×128 tile; the vLLM trace
shows a cooperative collective with a 128×128×256 tile. This template
difference is a testable explanation for some of the **38.8 ms** gap, but
different layouts and scheduling also matter; changing the tile alone has
not been benchmarked. The two alternating Xinfer grouped launches totaled
102.26 and 134.38 ms, respectively.

Finally, the current marked Xinfer range had 87.52 ms with no recorded GPU
kernel, copy, or memset, versus 3.22 ms in vLLM. With the fused-ReLU flag,
Xinfer's inactive time was 71.00 ms. Its memsets took 34.91 ms and wrote
7.09 GB, versus 5.43 ms and 1.18 GB in vLLM. These are distinct from the
named norm and expert kernel times; CUDA API waits overlap device work, so
their durations cannot be added to the gap. The existing zero-fill audit
should check whether full-output CUTLASS allocations can be uninitialized
only after proving every output element is written.

The fused residual-plus-RMSNorm kernel was implemented below. The
sorted-order expert intermediate was tested afterward and did not produce a
repeatable whole-prefill gain; see the follow-up at the end of this report.
The figures above overlap in memory bandwidth and launch scheduling, so
their apparent savings are not additive.

## Opt-in fused residual and RMSNorm

`XINFER_NEMOTRON_FUSED_RESIDUAL_NORM=1` enables a Nemotron-specific CUDA
kernel for BF16 prefill. A caller can also use
`NemotronHForCausalLM::set_fused_residual_norm_enabled(bool)` between forward
calls. The first layer norm and final residual remain on their original
paths; each other prefill layer combines the previous BF16 residual add with
the current FP32 RMSNorm. The kernel stores the rounded BF16 residual before
normalizing and reproduces Candle's 32- or 1,024-thread XOR reduction order.
Single-token decode is unchanged. The mode is disabled by default and
compiles its CUDA PTX on first use outside graph capture.

The focused CUDA test matched Candle's BF16 residual and normalized output
bytes at hidden widths 128, 2,688, and 4,480. On the real NVFP4 30B
checkpoint, alternating unfused/fused/unfused/fused runs on one loaded model
matched every prefix logit, every continuation logit hash, and the entire
portable Mamba state envelope byte for byte. The 8,448-token case used an
8,192-token call followed by a 256-token continued-prefill call, then 16
decode steps. The 9,538-token case used the fixed chat IDs in one call,
then 16 decode steps. A separate 2,048-token comparison matched all 65
saved logit vectors and the prefix state exactly.

| One-model prefill schedule | Unfused runs | Fused runs |
| --- | ---: | ---: |
| 8,192 + 256 continued tokens | 1,343.7 / 1,246.0 ms | 1,142.0 / 1,171.9 ms |
| 9,538 fixed chat tokens, one call | 1,385.5 / 1,370.0 ms | 1,248.4 / 1,245.0 ms |

The 9,538-token fused runs leave about 225–228 ms relative to the saved
1,020.3 ms vLLM range. That comparison spans separate sessions; it is a
directional remaining-gap estimate, not an isolated head-to-head result.

In a separate 9,538-token Nsight capture, 51 fused layer kernels totaled
50.37 ms. The original 52 adds, 53 norm casts in each direction, and 53
RMSNorms totaled about 179.6 ms in the prior trace. The captured grouped
expert GEMMs rose from 236.64 to 333.21 ms despite identical expert code,
and the marked whole-call range rose from 1,430.62 to 1,476.70 ms. That
capture illustrates shared-host clock/load variation; the alternating
one-model wall measurements and exact parity are the appropriate evidence
for this change. The fused trace is
`/tmp/nemotron-fused-residual-norm-9538.nsys-rep` with its SQLite export.

## Follow-up experiments on the remaining gap

The following experiments used the same 9,538 chat-token IDs and real NVFP4
30B checkpoint as the earlier captures. They were reverted after measurement;
the published engine path and its snapshot layout are unchanged. Whole-call
times are separate-process samples on a shared GB10 and are useful only as
coarse checks. The isolated CUDA traces provide the narrower kernel evidence.

| Experiment | Exactness or quality | 9,538-token prefill result |
| --- | --- | ---: |
| Keep the expert up-projection intermediate in expert-sorted BF16 order, then quantize it without scattering and gathering | All saved prefix/16 continuation logits and Mamba snapshot bytes matched the original path | 1,265.5 / 1,278.4 ms versus a fresh 1,272.9 ms control; no demonstrated gain |
| Replace precise SSD-output exponentials with `__expf` | Exported Mamba state changed; 249/257 teacher-forced top choices matched, maximum logit difference 4.1914 | 1,307.8 ms versus 1,278.1 ms control; slower and numerically unacceptable |
| Make the previously optional two-warp SSD chunk-prefix geometry the GB10 default | Prefix/16 continuation logits and Mamba snapshot matched byte for byte | 1,270.7 ms versus 1,269.3 ms for the reverse-bracketed four-warp control; no demonstrated whole-call gain |
| Double the custom attention query tile from 64 to 128 rows, retaining 32 keys | Batched-prefix and 2,048-row CUDA context oracles matched exactly; real checkpoint logits/state matched byte for byte | 1,342.0 ms versus recent 1,269–1,273 ms controls; slower |
| Double the custom attention key tile from 32 to 64 rows, retaining 64 queries | CUDA context differed from the paged BF16 oracle by up to 0.015625 at both 257 and 2,048 rows, exceeding the existing tolerance | Rejected before a full-checkpoint run |

At 2,048 rows, the isolated custom-attention kernel took **0.958 ms** for
64-query × 32-key, **1.008 ms** for 128-query × 32-key, and **1.192 ms** for
64-query × 64-key. The 128-query mode halved query CTAs but slowed the kernel;
the 64-key mode retained the same CTA count, slowed it by about 24%, and
changed the context. This rules out a direct tile-size change as the fix for
the earlier **~78 ms** six-layer attention gap. The comparison to vLLM's
128-query × 64-key trait is a clue about its broader implementation, not a
tile prescription for Xinfer's warp-local MMA kernel. Raw isolated traces are
`/tmp/nemotron-attn-{q64,q128,k64}-isolated.nsys-rep` on this host.

For the Mamba SSD path, a saved experimental tensor-core chunk-state trace
(`/tmp/nemotron-tensor-summary.sqlite`) spent **73.72 ms** in 46
`chunk_state_wmma` launches; a saved four-warp scalar summary trace spent
**73.59 ms** in its 46 summary launches. This does not establish a tensor-core
gain. The current full trace still spends about **76 ms** in scalar chunk
summary, **104 ms** in SSD output, **43 ms** in chunk prefix, and **28 ms** in
C×B. The next useful implementation should change how chunk state and output
are formed or reused, with a matched numerical and continued-prefill check;
simple warp, tile, and exponential substitutions have not closed the gap.

## C×B shared-scratch reuse

The opt-in SM121 WMMA C×B kernel now overwrites its dead BF16 input staging
tiles with the FP32 result after the final MMA. Its final loop barrier already
protects that reuse. For the usual K64 setting, static shared storage falls
from 32 KiB to 16 KiB per block; the ordered WMMA operations and BF16 output
rounding are unchanged. The one-warp C×B fallback and the model's other paths
are unchanged.

The focused CUDA SSD test matched every C×B BF16 word and FP32 output word
with nonzero carried state, zero decay, and a partial chunk. On the real 30B
checkpoint, all saved 9,538-token prefix and 16 continuation logits and the
entire exported Mamba state matched the prior kernel byte for byte. In
separate isolated 257-token Nsight captures, comparable warmed K64 C×B
launches fell from roughly 0.022 to 0.020 ms. These short captures suggest
a small kernel gain, not a measured whole-prefill speedup.

The two new real-checkpoint wall samples were 1,581 and 1,400 ms versus
recent 1,269–1,273 ms controls. They coincided with unstable host loading
(23 and 66 seconds) and one decode p50 rising to 35 ms from about 25 ms;
the 66-second load occurred as host available memory and free swap fell.
These samples cannot isolate the roughly millisecond-scale C×B improvement.
The next substantial gap reduction still requires a different SSD
summary/output or attention-kernel implementation.

## Skip unused C×B tiles

The SM121 `cb_gemm64` launch now computes three 64×64 tiles per 128-token
chunk instead of four. Its upper-right tile contains only products where the
source token is later than the output token. `ssd_output` already masks those
entries before loading C×B, so that tile never contributes to an output.
The remaining tiles keep the same WMMA operations and storage layout.

The focused CUDA oracle used a fresh C×B buffer filled with NaNs. The skipped
tile stayed poisoned, every consumed lower-triangle BF16 word matched the
full-grid reference, and FP32 output matched across K16/K64, 64/128 output
rows, direct/transposed starts, zero decay, and a partial chunk. On the real
NVFP4 30B checkpoint, a single 9,538-token prefill and 16 continuation steps
matched the saved full-grid logits (8,912,896 bytes) and exported Mamba
state (49,930,472 bytes) byte for byte.

This removes 25% of C×B thread blocks, not 25% of prefill time. Comparable
warmed K64 launches in separate 257-token Nsight captures had median
durations of 0.01987 ms before and 0.01891 ms after; this small difference
can include run-to-run clock variation. The correctly configured real-model
run took 1,250.8 ms, against recent full-grid samples around 1,269–1,273 ms
on the shared host. That is too little evidence to claim an end-to-end
speedup. The new isolated trace is
`/tmp/nemotron-ssd-triangular-isolated.nsys-rep`.

## Four query heads per attention block

`XINFER_NEMOTRON_ATTENTION_GQA_WARPS=1` is an opt-in layout for the custom
SM121 BF16 prefill kernel. Each of its four warps handles 16 rows of a
different query head; the four heads share one KV head and one staged K/V
tile. The ordinary mode assigns the four warps to consecutive rows of one
query head. The grouped mode retains each warp's QK, softmax, and PV order,
and supports both contiguous current K/V and paged carried prefixes.

The CUDA context test matched the ordinary kernel and the paged BF16 oracle
byte for byte with 257 rows, a 13-token carried prefix, partial tiles, and
two sequences. The same test matched at 2,048 and 9,538 rows. The real NVFP4
30B single-call 9,538-token prefix, 16 continuation logits, and exported
Mamba state matched the saved ordinary-mode bytes exactly.

| Isolated attention fixture | Ordinary custom kernel | Grouped query warps |
| --- | ---: | ---: |
| 2,048 rows, six warmed wall samples | 1.011–1.017 ms | 0.982–0.984 ms |
| 9,538 rows, six warmed wall samples | 22.70–23.21 ms | 22.34–22.81 ms |

These are same-process calls including cache writes and synchronization.
The real-model grouped run took 1,262.2 ms, whereas the preceding ordinary
run took 1,250.8 ms on the shared host. There is no demonstrated whole-model
gain. Grouping changes the query rows per block from 64 to 16 while reducing
the query-head blocks by four, so the total block count and aggregate K/V
loads are approximately unchanged. It mainly avoids masked work near the
causal frontier. A larger per-head row tile or a different attention kernel
is needed to address the measured 63–78 ms attention gap to vLLM.

## Prefix-state partitioning and fast attention exponentials

Two more changes were measured on the same GB10 and then reverted. Neither
changed the published inference path.

The first divided each Mamba chunk-prefix head's 64 state rows among two or
four independent CTAs. The hope was that more CTAs would hide the long
75-chunk state scan's memory latency. The 257-token CUDA recurrence test with
head width 64 passed, and the real 9,538-token prefix, 16 continuation logits,
and exported Mamba state matched the one-CTA path byte for byte. The saved
Nsight traces give the more useful result: over the 23 long-prefix kernel
launches, one CTA per head took **42.766 ms**, two took **50.398 ms**, and four
took **54.371 ms**. The whole-call wall times varied between 1,305 and
1,393 ms across those traces and cannot overturn the direct kernel result.
More CTAs replicated chunk-boundary control and memory traffic without enough
occupancy benefit. The traces are `/tmp/nemotron-prefix-p{1,2,4}-full9538.sqlite`.

The second change replaced the custom attention softmax's precise `expf`
with the GPU `__expf` intrinsic behind an experimental switch. On the
8,192-row BF16 CUDA fixture, the ordinary custom attention returned exactly
the paged-kernel BF16 context. The fast variant changed 200 of 33,554,432
BF16 context words, with maximum absolute difference **0.015625**. Six
same-process calls including cache writes and synchronization measured
**17.94–18.31 ms** for precise exponentials and **17.72–18.19 ms** for fast
ones. The samples ran in a fixed order, so this small difference is not an
isolated kernel gain. At 2,048 rows, 82 BF16 words changed with the same
maximum difference. The altered numerics without a convincing timing gain
do not justify using fast exponentials in the default path. The 8,192-row
CUDA oracle test passed for the unchanged precise path after the experiment.

These results narrow the next attention work to the QK/PV tile pipeline and
KV movement, rather than exponential precision. The larger Mamba opportunity
remains the chunk summary/output algorithm; duplicating the serial prefix
across more CTAs moves in the wrong direction.
