# Nemotron-H paged attention GEMM experiment on GB10

This is an opt-in specialization for the six BF16 attention layers of
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. Build with
`--features cuda,cutlass,nemotron-attention-gemm` using CUDA 13, then set
`XINFER_NEMOTRON_ATTENTION_GEMM=1`. The ordinary attention path remains the
default. The specialization is selected only for SM120+, 32 query heads,
two KV heads, width 128, BF16 paged K/V, no soft cap or sliding window, a
context of at least 4,096 tokens, and a prefill span of at least 1,024 tokens.
Decode and shorter prefills use the existing path.

The CUDA kernel derives from the Apache-2.0 licensed attention-rs paged
prefill kernel at revision `c0f19f2`. It changes the query tile from 32 to 64
rows and uses eight warps. Each warp retains the original 16-query-row QK
and PV accumulation order. Both paths write the same paged BF16 cache;
logical token positions and causal masking are unchanged. The Xinfer wrapper
checks shapes and dispatches only after the usual cache write.

## Measurements

The direct CUDA comparison used 8,192 tokens, two KV heads, 32 query heads,
width 128 and 16-token pages on the GB10. After the first iteration, the
original cache-write-plus-attention calls took 25.29–26.04 ms; the custom
calls took 19.20–20.66 ms. Their 8,192-token BF16 attention-context bytes
matched exactly. A separate 129-token test, including a two-sequence batch
with disjoint physical pages, also matched exactly.

On the real 30B checkpoint, one Nsight Systems capture of the exact
9,538-token prompt recorded 12 original attention-prefill launches totaling
185.25 ms; 12 custom launches totaled 130.98 ms. The kernels used 64
registers per thread in both traces. Whole-model release runs, with the same
8,192-token continued-prefill schedule and a 16 GiB host-memory watchdog:

| Prefix | Ordinary prefill | Custom prefill | Result |
| --- | ---: | ---: | --- |
| 2,048 repeated tokens | 472.65 ms | 491.90 ms before the length guard | All 33 logit rows byte-identical; guard now keeps ordinary path |
| 9,538 exact chat tokens | 2,176.26 ms | 2,080.39 ms | Prefix and one decode row byte-identical |
| 32,768 repeated tokens | 8,876.65 ms | 8,171.32 ms | Prefix and one decode row byte-identical |

These are single-process pairs on a shared host. The direct attention and
Nsight kernel measurements establish the local kernel gain; the whole-model
figures have run-to-run noise. The current 9,538-token whole-model time still
remains about twice the earlier vLLM first-token median of 1,036 ms on the
same token IDs. The attention specialization alone does not recover the lost
near-parity implementation: Mamba recurrence, expert gather/scatter, and
grouped expert GEMMs remain major costs.

## Validation and scope

The CUDA regression compares the new context with the original paged kernel
for non-tile-aligned length 129, length 8,192, and a two-sequence batch.
The real-model check compares BF16 checkpoint logits after continued prefill;
the 2,048-token run also compares 32 single-token continuation steps. The
kernel supports the existing cache envelope and does not change recurrent
state or snapshots. It is compiled only with the explicit Cargo feature
because the CUDA source targets SM120 and requires an NVCC toolchain that
supports it.
