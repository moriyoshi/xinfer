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
