# Nemotron-H GB10 prefill: route grouping and fused Mamba work

Measured 2026-10-14 on one NVIDIA GB10 using the local five-shard
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` checkpoint. Both old and new
Xinfer paths below use the same test harness and `cuda,cutlass` release build
settings, eager resident
experts, BF16 attention KV, and 8,192-token continued-prefill calls. This
change is based on `8494b26` on `epic/xinfer-session-enhancements`.

## Changes

- Nemotron prefill groups top-k assignments by expert with three CUDA kernels:
  count, prefix, and stable scatter. It keeps original assignment order within
  an expert and falls back to the generic sort outside the supported U32,
  at-most-256-expert range. `XINFER_NEMOTRON_ROUTE_GROUPING=0` selects the old
  sort for comparison. Decode routing is unchanged.
- For spans of at least 512 tokens, Mamba prefill now summarizes each 256-token
  interval as an affine state update, propagates only the chunk boundary
  states, and scans the chunks in parallel. The initial carried FP32 state is
  included, so nonzero-prefix continued prefill works. The exported Mamba
  snapshot envelope and layout are unchanged. Set
  `XINFER_NEMOTRON_CHUNKED_SCAN=0` to select the old sequential scan.
- The explicit host synchronization after each Mamba scan is removed. All
  dependent kernels and asynchronous frees use the same CUDA stream.
  `XINFER_NEMOTRON_PREFILL_SCAN_SYNC=1` restores the old barrier for comparison.
  Single-token decode is unchanged.
- One CUDA kernel now applies the Mamba gate, grouped RMSNorm, and BF16/F16/F32
  conversion for an entire prefill span. It follows Candle's vectorized FP32
  summation order for supported group widths and falls back to the original
  tensor operations otherwise. `XINFER_NEMOTRON_FUSED_GATE_NORM=0` selects the
  original epilogue.
- One CUDA kernel combines the top-k BF16/F16 expert output and shared expert
  using the original FP32 multiply and sum order, then rounds once to the
  model dtype. This replaces wide cast, multiply, reduce, add, and cast
  temporaries in both prefill and decode. Unsupported shapes or dtypes use the
  old Candle path; `XINFER_NEMOTRON_FUSED_MOE_COMBINE=0` is an opt-out.

## Paired real-checkpoint runs

The existing `real_30b_moe_prefill_and_decode_benchmark` test was built in
release mode with `RUSTFLAGS='-C target-cpu=native'`. Its first 512-token
forward warms the relevant kernels outside the timed region. Each row below is
one isolated process, not a multi-run median. Old-path runs used the
corresponding opt-outs as the implementation was built; the final build keeps
all four opt-outs. The watchdog required `MemAvailable >= 16 GiB`; the
accepted runs' minima were 89.9-94.4 GiB. One 38K run was discarded after an
unrelated linker reduced available memory to 21 GiB and inflated decode
latency. No unrelated GPU compute process was present at the start of the
accepted paired measurements.

| Input | Old prefill | New prefill | Reduction | Old/new decode p50 |
| --- | ---: | ---: | ---: | ---: |
| Repeated-token 2,048 prefix, 256 teacher-forced positions | 953.686 ms | 487.226 ms | 48.9% | 23.724/23.302 ms |
| Repeated-token 32,768 prefix, 32 teacher-forced positions | 19,161.584 ms | 8,810.901 ms | 54.0% | 29.278/27.857 ms |
| Exact earlier vLLM 9,538-token chat prefix | 4,307.719 ms | 2,262.821 ms | 47.5% | one decode step only |
| Exact earlier vLLM 38,004-token chat prefix | 20,197.760 ms | 10,734.540 ms | 46.9% | 4 new decode steps; 30.900 ms p50 |

The last two input JSON files reproduce the previously reported prompt token
counts and SHA-256 hashes exactly. Earlier vLLM normal-process first-token
medians on those same inputs were 1,036.0 and 5,149.4 ms, respectively. The
new Xinfer test times remain about 2.18× and 2.08× those vLLM medians. This
is an indicative cross-engine comparison: the Xinfer figures here use a direct
test model call, while vLLM uses `LLM.generate`, and they were measured in
separate runs. The matched old/new Xinfer figures use one harness and build
configuration; the baseline and final binaries were rebuilt as code changed.

At 2,048 tokens, isolated switches gave 894.427 ms with only route grouping,
814.349 ms with only the chunked scan, and 711.027 ms with only the host
barrier removed. These are single samples and their savings are not additive.

## Numerics and checks

The route-only run's complete 257 logit rows matched the old path byte for
byte. Removing only the host barrier matched the old path's first 33 rows byte
for byte. The fused Mamba gate/norm matched the nonfused optimized path byte
for byte over all 257 rows at 2,048 tokens and all 33 rows at 32,768 tokens.
The fused MoE combine also matched that optimized output byte for byte on
those same prefill and decode rows. A focused CUDA test checks its BF16 and
F16 rounding against Candle for nonaligned hidden widths. Chunking introduces
the numerical change. Against the old path,
the complete optimized 2,048-token run had 255/257 top-1 agreement, maximum
absolute logit difference 3.6875, and mean absolute difference 0.226788.
At 32,768 tokens, it had 33/33 agreement, maximum 1.9375, and mean 0.227698.
These are teacher-forced checks; free-running answer equivalence and task
quality have not been established.

An optimized 2,048-token prefill followed by 256 decode steps was repeated
in the same process and produced bit-identical prefix logits, decode logits,
and exported Mamba snapshot bytes for the same chunk schedule. Focused CUDA
tests cover stable route grouping through 8,192 rows, all assignments to one
expert, partial final Mamba chunks, nonzero initial state, continued prefill,
and deterministic repeat. The Mamba test compares the chunked output and
final state to the old sequential scan within explicit tolerances.

`cargo fmt --all -- --check`, `git diff --check`, and the complete CUDA/CUTLASS
library suite passed (372 passed, 2 ignored). Focused Clippy passed with
`clippy::never_loop` and `clippy::not_unsafe_ptr_arg_deref` suppressed; these
are existing deny-level errors in unrelated `api.rs` and `utils/graph.rs`.
Unrestricted `cargo clippy --lib --tests` also hits an unrelated
`tests/preprocess_parity.rs` constructor-signature error.

## Remaining GPU work

A bounded Nsight Systems capture of the final 9,538-token prefill recorded
4,159 GPU kernels and 2,017 ms of summed kernel time across a 2,237 ms GPU
span. The earlier vLLM profile recorded 1,017 ms of kernels for the same
token IDs. The largest final Xinfer components were the chunked Mamba scan
(315 ms) plus summary/prepare/prefix (162 ms), grouped expert CUTLASS GEMMs
(303 ms), expert gather/scatter (332 ms combined), and attention prefill
(171 ms). These kernel categories are indicative because the engine traces
were taken in separate runs. Closing the rest of the gap requires faster
recurrent and grouped-expert kernels, plus a better attention prefill path.

Raw paired logs, output logits, and the final `.nsys-rep`/SQLite trace are in
`/tmp/xinfer-nemotron-gap-results/` on the measurement host.

## Mamba scratch allocation follow-up

The chunk prefix now overwrites each chunk summary with its starting state
after reading the summary. This removes one FP32 scratch buffer: 64 MiB for
each 8,192-token Mamba span on this checkpoint. Mamba CUDA outputs also use
uninitialized allocations where the immediately following kernels write every
element on the same stream. This removes zero fills of convolution, decay,
scan, state, and normalized-output scratch; decode uses the same rule for its
single-token scratch. The carried state shape, snapshot bytes, chunk size, and
arithmetic order are unchanged.

A paired 9,538-token run with the same release binary settings and prompt
generator measured 2,201.228 ms before and 2,120.871 ms after (3.7% less).
The complete prefix-plus-one-step logit files were byte-identical. Additional
real-checkpoint runs measured 475.873 ms at 2,048 tokens and 8,566.769 ms at
32,768 tokens; the previous final-build runs recorded 487.226 and 8,810.901
ms. All 257 logit rows in the 2,048-token/256-step test and all 33 rows in the
32,768-token/32-step test were byte-identical to saved pre-change outputs.
Decode p50 was 23.773 and 28.241 ms, respectively, within the prior run-to-run
range. These are single-process samples on a shared host, so the small timing
differences are indicative rather than a controlled throughput distribution.

Nsight captures of the same 9,538-token input recorded 1,575 GPU memset
operations totaling 34.17 GB before, versus 1,207 totaling 20.69 GB after.
The captures used separate processes and profiler overhead inflated wall time;
their memset counts and bytes verify the intended traffic reduction. The new
profile and test logs are in `/tmp/xinfer-chunk-tuning/` on this host. Focused
CUDA tests cover sequential-scan parity, nonzero-state continued prefill,
single-token decode parity, and deterministic repeats.

## Fused expert activation follow-up

Nemotron-H now applies ReLU-squared to routed and shared expert activations in
one CUDA pass for contiguous BF16, F16, and F32 tensors. Unsupported layouts
and devices retain the Candle path. Set `XINFER_NEMOTRON_FUSED_RELU2=0` to use
the two original passes for a same-build comparison. A focused CUDA test checks
the output against Candle for routed and shared expert shapes at all three
dtypes.

For the exact 9,538-token checkpoint input, Nsight recorded 92 ReLU and 92
square launches totaling 136.16 ms before the fusion. The fused kernel used 92
launches totaling 60.90 ms, a 75.26 ms reduction in this kernel category. Two
interleaved same-build whole-model runs per setting were noisy: the original
path took 2,538/2,367 ms and the fused path 2,272/2,410 ms. Their medians are
2,452 and 2,341 ms, but the second pair reversed, so these samples do not
establish a precise end-to-end speedup. GPU activity from the expert gather
and scatter kernels also varied between traces. The two paths'
9,538-token prefix-plus-one-step logits matched byte for byte, as did all 257
logit rows of a 2,048-token prefix plus 256 decode steps. Decode latency did
not show a repeatable change. The new kernel removes an intermediate
activation tensor but does not address the larger Mamba scan, grouped GEMM,
attention, or expert gather/scatter costs.

## Mamba summary warp tuning

The 256-token chunk summary and chunk-boundary propagation now use four
warps per head; the state scan keeps eight. This reduces duplicate per-warp
loads in the summary without slowing the scan. Set
`XINFER_NEMOTRON_SUMMARY_WARPS=8` to select the former geometry. No recurrence
formula, chunk boundary, or state envelope changed.

Paired same-binary real-checkpoint prefill measurements on GB10, with the
same CUDA/CUTLASS release build and a 16 GiB `MemAvailable` watchdog:

| Prefix | Eight-warp summary | Four-warp summary | Difference |
| --- | ---: | ---: | ---: |
| 2,048 repeated sample IDs | 460.305 ms | 456.101 ms | 4.204 ms (0.9%) |
| 9,538 exact chat IDs | 2,077.599 ms | 2,038.209 ms | 39.390 ms (1.9%) |
| 32,768 repeated sample IDs | 8,272.938 ms | 8,142.517 ms | 130.421 ms (1.6%) |

Each row is one process per setting. A paired Nsight capture of the 9,538
IDs measured 104.263 to 66.764 ms for 46 chunk-summary launches and
21.780 to 19.708 ms for 46 prefix launches. The unchanged eight-warp scan
avoids the ten-millisecond regression seen when all three kernels used four
warps. The 9,538 and 32,768 runs' prefix-plus-one-step logits were byte
identical; the 2,048 runs' prefix plus 32 continuation logit rows were also
byte identical. Cross-setting exported 2,048-token Mamba state snapshots had
the same SHA-256 (`5374d6f751a87a408f029400e70f1c21cf8f4cf4842746dbcd7e5a4997d45dcf`).
Both settings passed repeated prefill, decode-logit, and state determinism
checks. The CUDA/CUTLASS library suite passed (373 passed, 2 ignored), as did
formatting and Clippy with the two existing unrelated deny-level lints
suppressed. Raw logs and profiles are in `/tmp/xinfer-chunk-tuning/` on the
measurement host. The end-to-end reduction is small and these are isolated
single-process samples, not a throughput distribution.

An experimental build with the existing `flashattn` feature was faster at
2,048 tokens (398.560 versus 464.269 ms), but every real-checkpoint logit
was NaN. A minimal paged BF16 probe wrote valid one-valued V-cache entries
yet returned a zero attention context. Nemotron-H construction now rejects
that feature on SM121 before loading model weights. The feature needs a
dependency fix and a real-checkpoint parity test before it can be used for
this model on GB10.
