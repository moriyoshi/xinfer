# Nemotron-H one-kernel prefill top-k on GB10

The opt-in `XINFER_NEMOTRON_TOPK_ONE_KERNEL=1` path selects experts for
128-expert, ungrouped sigmoid routing in one CUDA launch per MoE layer. It
reproduces attention-rs's descending 128-column bitonic comparison network,
including equal-score order. Score calculation, route weights, normalization,
and scaling still use the existing shared implementation. Single-token decode
continues to use Candle's last-dimension sorter, whose tie order differs.
All other layouts fall back to the original route.

The 30B NVFP4 checkpoint was
`/home/moriyoshi/src/xinfer-artifacts/nemotron-3-nano`. The guarded, paired
release benchmark used a single GB10, one prefill call, eager grouped NVFP4
experts, fused expert quant/scatter/finalizer, and the WMMA SSD output path.
The input was the benchmark's repeated fixed text, so these times must not be
compared directly with the separately measured vLLM chat prompt.

| Prefill / continuation | Generic top-k | One-kernel top-k | Check |
| --- | ---: | ---: | --- |
| 2,048 / 256 tokens | 372.480 ms | 370.175 ms | All 257 logit rows and exported prefix Mamba state byte-identical |
| 9,538 / 1 tokens | 1,638.602 ms | 1,614.340 ms | Both logit rows and exported prefix Mamba state byte-identical |

For the 2,048-token run, decode p50 was 24.206 ms with generic top-k and
24.108 ms with the prefill-only kernel. That difference is run noise; decode
routing is unchanged. The local timings are single pairs on a shared host.
The focused CUDA test compares selected IDs against generic bitonic sort for
1, 17, 257, and 8,192 rows using pseudo-random values, many ties, and NaNs.

The earlier one-call Nsight trace showed 644 bitonic sort launches across
23 MoE layers for 9,538 tokens (28 per layer), totaling about 22.9 ms of GPU
kernel time. This change collapses them to 23 launches. It removes a small
part of the remaining full-model gap; expert GEMM, attention, and Mamba SSD
remain larger contributors.
