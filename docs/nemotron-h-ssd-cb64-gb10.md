# Nemotron-H 64×64 WMMA C×B tile on GB10

The BF16 WMMA SSD path computes a 128×128 C×B matrix for each 128-token
chunk and Mamba group. Its original kernel launches one warp per 16×16 tile,
64 blocks per chunk and group. The optional 64×64 kernel uses a 4×4 warp array
and shared C/B tiles, reducing that to four blocks. Each warp executes the
same eight BF16 WMMA reduction steps; the packed C×B matrix and final output
match the original bit for bit in the CUDA partial-chunk, nonzero-state test.

Build with `--features cuda,cutlass,nemotron-mamba-ssd-wmma`, and set both
`XINFER_NEMOTRON_SSD_WMMA=1` and `XINFER_NEMOTRON_SSD_CB64=1`. The second
switch defaults to off. The recurrent state layout and snapshot envelope are
unchanged. Scratch size is unchanged; the new kernel uses about 20 KiB of
shared memory per block instead of one small warp tile.

## Real 30B NVFP4 checkpoint

The GB10 release benchmark used eager experts, BF16 attention KV, the fused
expert quant/scatter/finalizer, the opt-in fused expert ReLU quantizer, and
`nemotron-attention-gemm`. A guard required 64 GiB host `MemAvailable` and no
other GPU compute process at each start. The 32K prompt was four continued
8,192-token calls. All times exclude model load. These are individual runs on
a shared host, not a latency distribution.

| Prefix | 16×16 C×B | 64×64 C×B | Observation |
| --- | ---: | ---: | --- |
| 2,048, pair 1 | 373.472 ms | 374.727 ms | no resolved difference |
| 2,048, reverse pair | 373.418 ms | 371.420 ms | no resolved difference |
| 8,192, pair 1 | 1,323.635 ms | 1,314.926 ms | 0.66% faster |
| 8,192, reverse pair | 1,731.957 ms | 1,366.511 ms | host window noisy; not a reliable effect size |
| 32,768, pair 1 | 7,400.453 ms | 7,104.250 ms | 4.00% faster |
| 32,768, reverse pair | 7,419.464 ms | 7,022.480 ms | 5.35% faster |

The 8K Nsight traces put the C×B kernel at 71.62 ms for the 16×16 tile and
27.81 ms for the 64×64 tile, each over 46 launches including warmup. Those
kernel totals do not isolate the timed prefill. Full-model wall times were
more variable than the kernel time, especially at 8K. Prefill and one-step
continuation logits, plus exported Mamba state, were byte-identical between
tile choices at 2K, 8K, and 32K. No decode speedup is expected because the
single-token path does not use this SSD C×B kernel.
An eight-step 2K prefill/decode replay with the 64×64 path also produced
bit-identical logits and state on repetition.
