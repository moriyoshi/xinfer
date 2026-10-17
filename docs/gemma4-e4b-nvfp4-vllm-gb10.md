# Gemma 4 E4B NVFP4 on GB10: Xinfer and vLLM

Measured 2026-10-18. This compares the same [Gemma 4 E4B NVFP4 checkpoint](https://huggingface.co/Neural-ICE/Gemma-4-E4B-it-NVFP4), one sequence, and identical token IDs on one NVIDIA GB10. The checkpoint is revision `cccd4fa40a99a8d525e050f014ea21dc2b680995`. The Xinfer side uses fork revision `c05608857bab9e691fc8f377a7b165466d18b8a9`, `Gemma4ForCausalLM` in a release `cuda,cutlass` build with the default hardware NVFP4 decode path. The vLLM side uses NVIDIA's `nvcr.io/nvidia/vllm:26.05-py3` image (digest `sha256:654e563e727be1968487d453de0733fb074cb59adf424c779d0f9e7cfcb2b6b6`), vLLM `0.20.1+7124b12a.dev`, `modelopt_fp4`, tensor parallel 1, and `language_model_only=True`. The [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html) documents its text-only and KV-cache options.

Both engines use BF16 attention KV. Xinfer allocates BF16 KV directly. The published checkpoint asks vLLM for FP8 KV in `config.json` and `hf_quant_config.json`; a temporary metadata overlay sets `kv_cache_scheme` and `kv_cache_quant_algo` to `null` respectively. All vLLM model files other than those two metadata files are symlinks to the original checkpoint. The installed vLLM ModelOpt parser reports `kv_cache_quant_algo=None` for the overlay. vLLM allocated 24.14 GiB of KV cache for 450,825 token slots, consistent with E4B's 57,344 BF16 KV bytes per token. vLLM uses a 0.35 GPU memory utilization limit, one sequence, prefix caching disabled, chunked prefill enabled, a maximum model length and batch token limit of 8,256, and its default CUDA graph setting. It selected FlashInfer CUTLASS NVFP4 linear kernels and Triton attention; the latter was forced by Gemma 4's heterogeneous attention head dimensions.

The 40-token case is the model's chat-template prompt asking why the sky is blue. The 2,048 and 8,192-token cases repeat only its user-content tokens inside the same chat structure. The token-ID files are under `/tmp/xinfer-gemma4-vllm-bench/` on the measurement host, with SHA-256 values `997728dd562ef78858705633eca77f1143db0fa3ba5610def51b3278d3786491`, `73af67b622138682e8613e979f1cdaf218774407ac381f21e17fb7ab025f5262`, and `5222a75f5cf3aba368e85ee9d27976c20f25fee53b7b6c5e74580579cf9ac5be`. vLLM receives the raw IDs, so neither engine times tokenization.

Xinfer times one direct model prefill through the final logits copy and then 48 greedy single-token forwards, synchronizing each forward. vLLM uses `LLM.generate` after warmup: a one-token request measures first-token wall time. A separate 49-token request provides decode timing as `(full request wall time - that same request's first-token latency) / 48`; its internal first-to-last token timestamps corroborate the longer-context values. It uses greedy sampling and ignores EOS to force the full decode length. There are two isolated sweeps per engine in Xinfer–vLLM–Xinfer–vLLM order, each with three samples per size. The two APIs have different scheduler and sampling overhead. vLLM's default CUDA graphs are active while Xinfer's direct-model test has no graph capture, so the columns compare these practical execution paths rather than identical runtime machinery.

## Results

| Prompt tokens | Xinfer prefill | vLLM first token | Xinfer / vLLM prefill | Xinfer decode | vLLM decode | Xinfer / vLLM decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 43.3 ms | 61.1 ms | 0.71× | 36.0 ms/step | 23.8 ms/step | 1.51× |
| 2,048 | 487.6 ms | 199.2 ms | 2.45× | 38.6 ms/step | 24.5 ms/step | 1.58× |
| 8,192 | 2,192.9 ms | 903.1 ms | 2.43× | 41.0 ms/step | 25.4 ms/step | 1.61× |

Values are medians of six samples. First-token choices matched between engines at all three context lengths (`818`). The 40-token vLLM first-token samples ranged from 41.9 to 212.3 ms, so the apparent short-prompt Xinfer lead is less stable than the long-context gaps. The two sweep medians at 2,048 tokens were 470.1/526.0 ms for Xinfer prefill and 200.4/186.9 ms for vLLM; at 8,192 they were 2,179.9/2,231.2 ms and 916.3/900.9 ms. Xinfer decode sweep medians at 2,048 were 38.4/41.6 ms versus vLLM's 24.4/24.6 ms. These ranges expose run-to-run drift on the shared host.

The GPU was otherwise idle for these sweeps. A one-second watchdog observed minimum `MemAvailable` of 94.4 GiB during Xinfer and 35.4 GiB during vLLM. The watchdog would stop a run below 16 GiB, and the vLLM container had an 80 GiB memory limit. Model loading and vLLM compilation/autotuning/graph capture were excluded from request timings. vLLM logged a warning that global NVFP4 scales differ between some parallel projections, which may reduce accuracy. The 40-token greedy continuations shared only their first five tokens across engines, so these are matched-workload latency results, not output-parity results.

### vLLM eager control

One further vLLM sweep used `enforce_eager=True` with the same weights, BF16 KV, token IDs, and limits. In this vLLM build that setting disables **both** `torch.compile` and CUDA graphs and changes the available fusions, so the difference cannot be assigned to graph capture alone.

| Prompt tokens | Eager first token | Eager decode, post-first wall |
| ---: | ---: | ---: |
| 40 | 157.2 ms | 105.0 ms/step |
| 2,048 | 253.0 ms | 100.7 ms/step |
| 8,192 | 1,058.5 ms | 92.8 ms/step |

These are three-sample medians. vLLM's normal compiled/graph mode is therefore material to its decode advantage here. The main table compares the production-like default mode, while the eager control shows why direct model calls should not be equated to an uncompiled vLLM path.

## Xinfer CUDA profile and opt-in GEMV follow-up

An isolated Nsight Systems trace of the same Xinfer real-checkpoint test used `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`. The large vocabulary-projection GEMV marks the end of each forward, allowing the three 2,048-token and three 8,192-token prefill intervals to be separated from decode. The figures below are from one representative interval per size; profiling adds overhead, so compare the kernel components within the trace rather than treating its wall time as a new unprofiled benchmark.

| Xinfer prefill | Wall interval | GPU kernel time | Attention | FP4 CUTLASS GEMM | RMS norms | BF16 copies and casts | Other kernels and idle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 tokens | 464 ms | 413 ms | 30 ms | 70 ms | 50 ms | 146 ms | 168 ms |
| 8,192 tokens | 2,157 ms | 1,944 ms | 313 ms | 278 ms | 230 ms | 601 ms | 735 ms |

The copy-and-cast column includes `ucopy_bf16`, `cast_bf16_f32`, and `cast_f32_bf16`; it is 227.7 + 193.9 + 179.4 ms at 8,192 tokens. The remaining GPU work there includes BF16 multiply (156 ms), add (92 ms), GELU (89 ms), activation quantization (106 ms), and RoPE (41 ms). These components are additive kernel durations, not exclusive attribution to specific source lines. The 42-layer model has seven full-attention and 35 sliding-attention layers, matching the 7 + 35 prefill attention launches. Attention is material but cannot alone explain the approximately 1.29-second first-token difference from vLLM at 8,192 tokens. The trace establishes Xinfer's costs; a corresponding vLLM GPU trace is still needed to assign its speedup to particular fusions or kernels.

In the full Xinfer trace, a cuBLAS BF16 GEMV ran 443 times at a median 8.62 ms and consumed 18.8% of recorded GPU kernel time. There are 432 measured decode steps; this shape is consistent with the tied 262,144-row vocabulary head. The existing `XINFER_GEMMA4_BF16_GEMV=1` option replaces eligible decode GEMVs with Xinfer's row-parallel kernel. On the same NVFP4 checkpoint, an additional isolated three-sample run measured decode medians of 32.12, 34.84, and 37.12 ms per step at 40, 2,048, and 8,192 tokens. The original default-mode six-sample medians were 36.00, 38.64, and 40.96 ms. The opt-in mode closes about 4 ms of decode time but remains slower than default vLLM's 23.81, 24.50, and 25.44 ms. The first-token choice stayed 818; this run did not establish full continuation parity across GEMV modes, and BF16 accumulation order differs.

## Reproduction

`tests/prepare_gemma4_vllm_bench.py` builds the exact token streams and a two-file BF16-KV metadata overlay from the checkpoint and the 40-token reference. It leaves the weight shards untouched; the symlinks in the overlay resolve when the original checkpoint is mounted at `/weights` in Docker. Xinfer's opt-in test is `bench_real_e4b_nvfp4_single_sequence` in `tests/gemma4_shared_kv.rs`.

```sh
MODEL=/path/to/Gemma-4-E4B-it-NVFP4
BENCH=/tmp/xinfer-gemma4-vllm-bench
python3 tests/prepare_gemma4_vllm_bench.py \
  --checkpoint "$MODEL" --base-tokens /path/to/gemma4/ref/tokens.json \
  --output "$BENCH"
XINFER_NVFP4_HW_DECODE=1 XINFER_GEMMA4_NVFP4_CHECKPOINT="$MODEL" \
XINFER_GEMMA4_BENCH_DIR="$BENCH" \
cargo test --release --features cuda,cutlass --test gemma4_shared_kv \
  bench_real_e4b_nvfp4_single_sequence -- --ignored --nocapture
docker run --rm --gpus all --ipc=host --shm-size=4g \
  --memory=80g --memory-swap=80g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL:/weights:ro" -v "$BENCH/model:/model:ro" \
  -v "$BENCH:/bench" -v "$PWD/tests:/xinfer-tests:ro" \
  --entrypoint python3 nvcr.io/nvidia/vllm:26.05-py3 \
  /xinfer-tests/bench_gemma4_vllm.py --input /bench --model /model
```

The measured runs used the same vLLM request loop in a temporary script, with a one-second 16 GiB `MemAvailable` watchdog. Its raw JSON/logs are under `/tmp/xinfer-gemma4-vllm-bench/` on this host. The `config.json` and `hf_quant_config.json` overlay SHA-256 values are `9908dddb1f1acd268c3c675a9803df609297c0b1fd7e7faf90ace3e8ac007796` and `582ec30054b7a7618278c936f7bc9724dd773e5d6accabdcd599bc0eab788511`. For the eager control, add `--eager` to the Python command. The command above omits the host-memory watchdog; run it on an idle GPU with sufficient available host RAM.
