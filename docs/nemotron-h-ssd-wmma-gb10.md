# Nemotron-H Mamba SSD WMMA experiment on GB10

The optional `nemotron-mamba-ssd-wmma` Cargo feature builds a GB10 (`sm_121`)
BF16 tensor-core output path for the 64-head, 64-channel, 128-state Nemotron-3
Mamba mixer. Set `XINFER_NEMOTRON_SSD_WMMA=1` to use it. Other configurations
continue to use the existing scalar scan. The default path is unchanged.

The existing causal convolution, FP32 chunk summary, carried-state prefix,
gated norm, and output projection remain in place. Each 128-token SSD chunk
computes a shared `C × B` matrix for its eight-head group, multiplies its
causally decayed weights by `X` with BF16 WMMA, and adds the carried-state
term. Zero decay resets are tracked explicitly. The final FP32 state comes
from the same affine chunk prefix, so continued prefill and the existing
Mamba snapshot envelope are supported. The kernel's BF16 intermediate values
change logits and state values relative to the scalar path. Decode is
unchanged. `nvdisasm` shows `HMMA.16816.F32.BF16` in the compiled cubin.

## Real-checkpoint measurements

Checkpoint: `/home/moriyoshi/src/xinfer-artifacts/nemotron-3-nano`,
Nemotron-3-Nano-30B-A3B NVFP4. One GB10; no other GPU compute process was
running. The release benchmark used `cuda,cutlass,nemotron-attention-gemm,`
`nemotron-expert-fused-quant,nemotron-mamba-ssd-wmma`, all three fused expert
environment switches, and `RUSTFLAGS='-C target-cpu=native'`. Timings include
the complete model prefill, excluding model load. The prompt is the benchmark's
repeated sentence, so these numbers are for a controlled kernel comparison,
not a serving distribution.

| Tokens and call schedule | Scalar scan | SSD WMMA | Improvement |
| --- | ---: | ---: | ---: |
| 2,048 in one call | 401.6 ms | 361.6 ms | 10.0% |
| 32,768 in four 8,192-token calls | 7,264.7 / 7,275.4 ms | 6,811.0 / 6,792.0 ms | 6.4% paired mean |
| 32,768 in one call | 9,036.2 ms | 7,582.2 ms | 16.1% |

The production-like four-call comparison was repeated in reverse order.
At 2,048 tokens, Nsight Systems reported 381.65 ms of GPU kernels for the
scalar path and 338.98 ms for SSD WMMA. The 23 scalar chunk-scan kernels took
72.71 ms; the 23 SSD output kernels took 23.61 ms, plus 7.82 ms for `C × B`,
5.21 ms for state transpose, and 1.36 ms for decay prefix. More 128-token
chunks also made the FP32 state-prefix kernels take 8.96 ms versus 2.89 ms.

For the 2,048-token prefix and 256 teacher-forced continuation positions,
top-1 logits agreed at 255/257 positions; maximum absolute logit difference
was 3.1875. The 32,768-token last-prefix top choice agreed in both tested
call schedules. Repeating the same 2,048-token prefill and all 256 decode
steps produced bit-identical logits and snapshot bytes. A repeated four-call
32,768-token prefill plus 16 decode steps was also bit-identical. A 257-token
CUDA recurrence test covers nonzero carried state, zero decay, a partial final
chunk, and exact repeat output.

## Limits

The 128-token chunks double FP32 summary scratch compared with the default
256-token chunks. Including the `C × B`, transposed state, and decay buffers,
the extra logical scratch over the scalar path is about 148 MiB for each
8,192-token call, or 592 MiB for one 32,768-token call. This is transient per
Mamba layer; allocator reservation and physical VRAM peak were not measured.
The native kernel is built for `sm_121` and requires CUDA toolkit support for
that target. Large spans beyond its signed 32-bit transpose grid are rejected
before SSD scratch allocation. Other models and GPUs use the original path.

The 6.4% gain on the 8,192-token schedule is useful but does not close the
remaining Xinfer versus vLLM prefill gap. These teacher-forced results do not
establish free-running quality equivalence.
