# Nemotron-3-Nano NVFP4 prefill on GB10: Xinfer and vLLM

Measured 2026-10-14 on one NVIDIA GB10 (driver 580.159.03). This compares the local five-shard `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` checkpoint, one sequence, batch size one, and the same four token-ID streams. The local checkpoint's `config.json` SHA-256 is `f4fcbaaaed3ee215f86205a7600636cd930bde069f2b603d7ff908c046a0cd96` and its original `hf_quant_config.json` SHA-256 is `37ff9e85d04311916a38ff7bc28cae1972dfdc54d39da8439d3f85204b9e8fbf`.

## Setup and timing contract

- **Xinfer:** fork revision `1ee989750cc57ce13d3620a6e7142dbb9871599f`, release build with `cuda,cutlass`, all experts resident, BF16 attention KV. The pinned `llmbda-xinfer-tools hybrid-suspend-cost MODEL OUT 3 0 64 256 1024` probe calls one `Session::extend` over each full prompt. Its `prefill_ms` includes the forward and final logits transfer, but excludes checkpoint export and publication.
- **vLLM:** NVIDIA `nvcr.io/nvidia/vllm:26.05-py3` (image digest `sha256:654e563e727be1968487d453de0733fb074cb59adf424c779d0f9e7cfcb2b6b6`), vLLM `0.20.1+7124b12a.dev`, FlashInfer `0.6.10+cf494fca.nv26.05.cu132.50619265`. `modelopt_fp4`, BF16 model and attention KV, tensor parallel 1, `max_num_seqs=1`, `max_model_len=max_num_batched_tokens=39000`, prefix caching off, chunked prefill enabled, eager mode, `gpu_memory_utilization=0.35`. Set `VLLM_USE_FLASHINFER_MOE_FP4=1` and `VLLM_FLASHINFER_MOE_BACKEND=throughput`. A temporary metadata overlay changed only `quantization.kv_cache_quant_algo` from `FP8` to `null`; all checkpoint files remained symlinks to the same weights. With `kv_cache_dtype=auto`, this removes ModelOpt's FP8 KV override and uses the BF16 model dtype. vLLM's 15.46 GiB cache allocation reported 2,184,000 token slots, consistent with BF16 attention KV plus Mamba state.
- **Prompts:** the Nemotron chat template, one tool, a fixed user question, and 0/64/256/1024 numbered system rules. Their token counts were 278/2504/9538/38004. The token-ID JSON SHA-256 values were `ef58ceba47552019edb04c99eafac08e1a84a359408362c56ee42c868f05195c`, `f54f0c06e81d46e2ebbab8655783922cada4119125323fdc49aef8ec1c77952a`, `a32fe77947d6b586208bee6480b79bff28bd6c7a5ecf132b479985ef1195cf86`, and `1d5608eca01d1d5e252fd4188db097521dc6882195f6e0b1fb0a6bc7aac6dbfa`. The vLLM tokenizer reproduced these exact IDs.
- **vLLM timing:** `LLM.generate` with `max_tokens=1`, greedy sampling, measured as blocking client wall time after engine startup and a warmup. Decode is estimated from a separate `max_tokens=33` request: `(33-token request time - one-token request time)/32`. Each size has three samples. Xinfer's direct model call and vLLM's scheduled first-token request are similar workload measures, though their API and scheduler overhead differ.
- **Shared host:** both sweeps ran alone on the GPU behind a one-second `MemAvailable` watchdog that would kill the run below 16 GiB. The observed minima were 62.1 GiB for Xinfer and 51.9 GiB across the vLLM attempts. CPU activity elsewhere on the host and clock variation were not controlled.

The vLLM log selected FlashInfer CUTLASS NVFP4 dense GEMM and grouped MoE, Triton Mamba SSU, and FlashAttention 2. The checkpoint has 23 Mamba, 23 MoE, and 6 attention layers. NVIDIA's [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) documents its vLLM environment settings; the [26.05 container notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-05.html) document the GB10 memory and Nemotron sequence limits.

## Three-sample medians

| Prompt tokens | Xinfer prefill | Xinfer ms/token | vLLM first token | vLLM ms/token | Xinfer/vLLM | vLLM decode ms/step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 278 | 196.9 ms | 0.708 | 101.3 ms | 0.364 | 1.94× | 24.9 |
| 2,504 | 1,236.3 ms | 0.494 | 305.7 ms | 0.122 | 4.04× | 24.9 |
| 9,538 | 4,757.1 ms | 0.499 | 1,036.0 ms | 0.109 | 4.59× | 27.2 |
| 38,004 | 22,607.7 ms | 0.595 | 5,149.4 ms | 0.135 | 4.39× | 26.0 |

Xinfer's first 278-token sample included first-use setup (1,042 ms); the other two were 181 and 197 ms. The earlier llmbda three-sample medians were 200/1710/6213/23336 ms. The fresh mid-length Xinfer run was faster; the long result was close. These are latency measurements, not model-output or numerical-parity tests between engines.

## Nsight Systems at 9,538 tokens

Nsight Systems `2025.3.2` profiled one warmed prefill per engine. Xinfer used the same pinned executable and a fresh model load. Its prefill interval was the GPU burst after loading and before the first final-logits device-to-host copy; the reported probe time was 5,190.6 ms. vLLM used a `BENCH_PREFILL_9538` NVTX range; its measured interval was 1,033.3 ms. For **profiling only**, `VLLM_ENABLE_V1_MULTIPROCESSING=0` put vLLM's GPU worker in the profiled process because this Nsight build did not capture CUDA events from vLLM's spawned worker. The timing table above uses vLLM's normal multiprocess mode. The in-process profile selected the same model kernels and gave 1,033 ms for its prefill.

| Identifiable GPU work in one prefill | Xinfer kernel time / launches | vLLM kernel time / launches |
| --- | ---: | ---: |
| Mamba scan or chunked SSD, including prepare/conv | 920 ms / 46 | 140 ms / 138 |
| Grouped expert NVFP4 GEMMs | 323 ms / 46 | 198 ms / 46 |
| Selected MoE routing and row movement kernels | 463 ms / 3,864 | 26 ms / 138 |
| Main attention prefill kernel | 181 ms / 6 | 62 ms / 6 |
| Dense NVFP4 CUTLASS GEMMs | 76 ms / 80 | 79 ms / 80 |
| BF16 CUTLASS GEMMs | 59 ms / 29 | 61 ms / 29 |
| **All GPU kernels, including other pointwise work** | **3,306 ms / 5,449** | **1,017 ms / 1,207** |

These rows identify named kernels; they are not an additive layer-time partition. For example, pointwise casts and route weighting appear in the all-kernel total, and the two MoE implementations fuse different work. Xinfer's 3,772 `bitonic_sort_kernel` launches alone took only 28.6 ms on the GPU, but account for 69% of its launch count. Its 23 `nemotron_mamba_scan` kernels took 879.5 ms, versus vLLM's six SSD/conv kernels per Mamba layer totaling about 140 ms. Xinfer's 46 expert gather and 46 scatter kernels took 249 and 186 ms; vLLM's 23 `expandInputRowsKernel` launches took 21.6 ms. The main attention kernel took 181 ms in Xinfer versus 62 ms in vLLM.

Xinfer called `cuMemAllocAsync` 1,880 times and `cuStreamSynchronize` 23 times during this prefill. The source synchronizes immediately after each Mamba scan. Nsight attributed 2.34 s and 2.02 s of **CUDA API call time** respectively; those sums overlap GPU work and must not be added to the kernel times or treated as fully recoverable savings. vLLM's profiled range had 5 short stream synchronizations totaling 0.03 ms. Xinfer made no device-to-host copies inside the prefill interval; its first copy was the final logits read. The profiled Xinfer interval exceeded its summed kernel durations by about 1.88 s, whereas the vLLM interval nearly matched its summed kernel durations. This is an indicative host/launch gap, not a causal subtraction across independent runs.

## Ranked Xinfer work

1. **Reduce MoE routing and allocation churn.** Replace the full Candle bitonic sort of all top-6 assignments with a bounded expert-count/prefix-sum/permutation path, reuse routing/GEMM scratch, and fuse or coalesce gather, activation, and weighted scatter where exact routes and output numerics permit. This targets 3,772 sort launches, 1,880 async allocations, and over 400 ms in gather/scatter GPU work. Measure wall time and exact routed-output parity after each step.
2. **Use a chunked Mamba-2 SSD prefill.** The existing fused scan is much better than token-at-a-time execution, but its scan/prepare pair costs 920 ms at 9.5K versus vLLM's 140 ms SSD/conv path. Preserve continued prefill at nonzero prefix positions and the exported state envelope; separately verify deterministic logits/state for identical chunk schedules and quantify numerical differences.
3. **Tune grouped NVFP4 experts.** The 46 expert GEMMs take 323 ms versus 198 ms in vLLM, even after both engines use device-side routing. Test FlashInfer-like CUTLASS tile choices and fused expert activation/finalization on sm_121, keeping packed weights, top-6 selection, and exact fallback behavior.
4. **Improve attention prefill at long spans.** Six main attention launches cost 181 ms versus 62 ms. Try a contiguous or larger-page prefill kernel while preserving paged BF16 KV state and decode. Dense NVFP4 and BF16 GEMMs were similar in this trace and are lower priority.

The raw JSON, logs, temporary vLLM prompt harness, and `.nsys-rep`/SQLite traces are under `/tmp/xinfer-nemotron-prefill/` on this host. They are local benchmark artifacts; the results and reproduction contract above are retained here.
