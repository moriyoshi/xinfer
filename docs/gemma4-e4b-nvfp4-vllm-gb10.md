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

The copy-and-cast column includes `ucopy_bf16`, `cast_bf16_f32`, and `cast_f32_bf16`; it is 227.7 + 193.9 + 179.4 ms at 8,192 tokens. The remaining GPU work there includes BF16 multiply (156 ms), add (92 ms), GELU (89 ms), activation quantization (106 ms), and RoPE (41 ms). These components are additive kernel durations, not exclusive attribution to specific source lines. The 42-layer model has seven full-attention and 35 sliding-attention layers, matching the 7 + 35 prefill attention launches. Attention is material but cannot alone explain the approximately 1.29-second first-token difference from vLLM at 8,192 tokens.

In the full Xinfer trace, a cuBLAS BF16 GEMV ran 443 times at a median 8.62 ms and consumed 18.8% of recorded GPU kernel time. There are 432 measured decode steps; this shape is consistent with the tied 262,144-row vocabulary head. The existing `XINFER_GEMMA4_BF16_GEMV=1` option replaces eligible decode GEMVs with Xinfer's row-parallel kernel. On the same NVFP4 checkpoint, an additional isolated three-sample run measured decode medians of 32.12, 34.84, and 37.12 ms per step at 40, 2,048, and 8,192 tokens. The original default-mode six-sample medians were 36.00, 38.64, and 40.96 ms. The opt-in mode closes about 4 ms of decode time but remains slower than default vLLM's 23.81, 24.50, and 25.44 ms. The first-token choice stayed 818; this run did not establish full continuation parity across GEMV modes, and BF16 accumulation order differs.

## Fused Gemma 4 follow-up

The `XINFER_GEMMA4_FUSED_GELU` path applies gated GELU directly to the merged NVFP4 gate/up output. `XINFER_GEMMA4_FUSED_RMSNORM` uses BF16-input/FP32-weight CUDA RMSNorm kernels, combines post-attention normalization with residual addition and the next normalization, and combines the other normalization/residual additions. Both are enabled by default when constructing BF16 NVFP4 Gemma 4 on CUDA; set either variable to `0` to disable that fusion. Explicit `1` retains the opt-in behavior for other supported Gemma 4 configurations. The kernels are loaded before CUDA graph capture and preserve the BF16 roundings used by the original path.

On the same GB10 and checkpoint, a three-sample isolated sweep with both options and the pre-existing `XINFER_GEMMA4_BF16_GEMV=1` option gave:

| Prompt tokens | Fused Xinfer prefill | Fused Xinfer decode | Original Xinfer prefill | Original Xinfer decode | vLLM first token | vLLM decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 35.4 ms | 30.5 ms/step | 43.3 ms | 36.0 ms/step | 61.1 ms | 23.8 ms/step |
| 2,048 | 284.9 ms | 33.3 ms/step | 487.6 ms | 38.6 ms/step | 199.2 ms | 24.5 ms/step |
| 8,192 | 1,382.4 ms | 35.6 ms/step | 2,192.9 ms | 41.0 ms/step | 903.1 ms | 25.4 ms/step |

The fused prefill is 37% faster at 8,192 tokens. It remains 1.53× vLLM first-token latency; fused decode remains 1.40× vLLM. These are adjacent Xinfer measurements compared with the earlier vLLM sweep, not a simultaneous cross-engine rerun. A separate real-checkpoint parity test compared models with and without the two new fusions on 40, 2,048, and 8,192-token prefixes and 48 teacher-forced continuation positions at each length. Maximum absolute logit difference was **zero** and top-1 agreed at all 147 positions. A second run verified the default selection against explicit `0` opt-out. The GEMV option was not part of those parity comparisons.

A worker-level vLLM CUDA trace for an 8,192-token prompt records 286 ms of attention, 283 ms of FP4 GEMM, 93 ms of separate FP4 activation quantization, and 97 ms of fused gated GELU/FP4 quantization. The corresponding **prefill-only** Xinfer Nsight interval records 311 ms of attention and 278 ms of CUTLASS FP4 GEMM. An earlier extraction mistakenly included decode kernels after prefill and overstated Xinfer GEMM as 349 ms. The corrected GEMM comparison shows that FP4 matrix multiplication is not the main remaining lever for this checkpoint. Gemma E4B has its MoE block disabled, so expert dispatch is not involved.

In that Xinfer prefill interval, 301 activation quantizations take 109 ms, scale swizzles 10 ms, 42 gated GELU launches 101 ms, norm/add kernels 136 ms, other FP32 RMSNorm 40 ms, BF16/FP32 casts 80 ms, BF16 multiply 36 ms, and RoPE 42 ms. The Q/K normalization path alone accounts for much of the FP32 RMSNorm and cast work: 42 large per-head Q norms take about 32 ms, with additional K norms and casts before and after RoPE. A Gemma-specific Q/K norm plus RoPE kernel that reads BF16 projections, computes the existing FP32 normalization/rotation, and writes BF16 Q/K is a stronger isolated target than fusing gated GELU with FP4 quantization. Broader residual/norm/cast fusion is also material. vLLM's generated kernels combine several of those operations. Different profilers and execution paths prevent subtracting the kernel sums as an exact cross-engine savings estimate; decode also benefits from vLLM's compiled CUDA graphs, which Xinfer's direct-model benchmark does not use.

To reproduce the measured path, add `XINFER_GEMMA4_BF16_GEMV=1` to the Xinfer benchmark command below; the two new fusions are automatic for this checkpoint. `cargo test --release --features cuda,cutlass --test gemma4_shared_kv real_e4b_nvfp4_default_fusions_match_opt_out -- --ignored --nocapture` runs the real-checkpoint default/opt-out parity check with the same checkpoint and token-file environment variables. The Nsight and vLLM trace files are under `/tmp/xinfer-gemma4-vllm-bench/` on the measurement host.

## Fused Q/K norm and RoPE

The Gemma 4 BF16 NVFP4 CUDA path now reads projected BF16 Q/K, performs the existing per-head RMSNorm and RoPE in FP32, and writes BF16 attention inputs in one kernel. Shared-KV layers use its query-only mode, avoiding the scratch K used by the ordinary RoPE call. It is enabled by default for this configuration; set `XINFER_GEMMA4_FUSED_QK_ROPE=0` to use the original path. The fusion applies only when the model has full-width FP32 RoPE tables and per-head FP32 normalization weights; other layouts retain the ordinary path. The module is prepared during model construction for CUDA graph capture.

The kernel uses Candle's 32-thread reduction order for head widths below 1024. A first implementation used a different reduction order and changed one top choice at the 40-token prefix. With the matching reduction, the real E4B NVFP4 checkpoint produced **bit-identical logits** between opt-out and fused models for 40, 2,048, and 8,192-token prefixes and 48 teacher-forced continuation positions after each prefix: maximum absolute logit difference zero and top-1 agreement 147/147. The test is `real_e4b_nvfp4_fused_qk_rope_matches_baseline` in `tests/gemma4_shared_kv.rs`. A focused CUDA test also compares the kernel with FP32 Candle RMSNorm plus attention-rs RoPE for interleaved and non-interleaved layouts and query-only operation.

In an isolated GB10 sweep on the same checkpoint with the existing fused GELU, RMSNorm, and BF16 GEMV paths enabled, six samples per mode in baseline/fused/fused/baseline order gave:

| Prompt tokens | Q/K opt-out prefill | Fused Q/K prefill | Prefill gain | Q/K opt-out decode | Fused Q/K decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 34.86 ms | 33.18 ms | 4.8% | 30.10 ms/step | 28.96 ms/step |
| 2,048 | 288.03 ms | 255.69 ms | 11.2% | 32.98 ms/step | 31.55 ms/step |
| 8,192 | 1,388.32 ms | 1,253.98 ms | 9.7% | 35.31 ms/step | 33.96 ms/step |

The 8K fused prefill remains 1.39× the earlier vLLM first-token median of 903.1 ms, with 351 ms remaining gap. These vLLM and fused Xinfer sweeps were on the same checkpoint and token IDs but at different times. An isolated 8K Q/K-only control measured median 10.94 ms for the original cast/norm/RoPE/cast sequence and 2.90 ms for the fused kernel, using an 8-query-head/2-KV-head/256-width shape. The real model's savings are smaller than 42 times this synthetic result because head shapes and other work differ.

An Nsight Systems interval for one fused 8K prefill records 1,057 ms summed GPU kernel time versus 1,183 ms in the comparable opt-out trace. The fused Q/K kernel takes 31 ms across 42 launches. FP4 CUTLASS GEMM takes 278 ms and attention 311 ms, both already close to the vLLM trace's 283 and 286 ms. The next largest Xinfer components are activation quantization 103 ms, gated GELU 98 ms, fused RMSNorm/add 67 ms, post-attention/pre-feedforward normalization 43 ms, and BF16 multiplication 36 ms. The trace and per-mode benchmark logs are under `/tmp/xinfer-gemma4-qk-bench/` on this host.

The same 8K interval has 737 host-to-device copies totaling only 9,356 bytes; 603 are four-byte copies. Two four-byte scalar uploads occur in each of the 301 NVFP4 linear calls at `attention-rs/src/nvfp4_linear.rs` when calibrated input scales are used. Nsight reports 211 ms summed CPU API time for the 737 copies and 0.385 ms of GPU transfer time. The API duration overlaps GPU work and is **not** an additive 211 ms saving opportunity. A resident per-linear GPU scale pair, passed through an attention-rs NVFP4 API, is a plausible next cross-crate lever to test before another GEMM rewrite. The other remaining Xinfer candidates are the residual/norm chain and prefill host launch overhead. The four paired sweeps recorded no competing GPU process; minimum `MemAvailable` after the prior fuzz process exited was 99.1 GiB.

## Fused layer output scale

Each Gemma 4 decoder layer multiplies its BF16 output by a BF16 `layer_scalar` after the final RMSNorm and residual addition. A Xinfer-owned CUDA kernel now performs that multiply within the final RMSNorm/add launch, preserving the separate BF16 rounding of the norm and residual sum. It is enabled by default for the BF16 NVFP4 CUDA configuration when fused RMSNorm is enabled; set `XINFER_GEMMA4_FUSED_LAYER_SCALE=0` to use the separate multiply. Other dtypes, devices, and unsupported tensor layouts use the existing operations. The real checkpoint yielded bit-identical logits for 40, 2,048, and 8,192-token prefixes and 48 teacher-forced continuation positions after each prefix (147/147 top-1). The focused CUDA test also compares every output BF16 word with separate RMSNorm/add/multiply.

With no other GPU compute process, two six-sample sweeps in baseline/fused/fused/baseline order used the same checkpoint, token IDs, default Q/K and GELU/RMSNorm fusions, hardware NVFP4 decode, and `XINFER_GEMMA4_BF16_GEMV=1`. The baseline retained fused RMSNorm/add and disabled only the final scale fusion:

| Prompt tokens | Separate scale prefill | Fused scale prefill | Separate scale decode | Fused scale decode |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 35.15 ms | 35.46 ms | 31.24 ms/step | 29.93 ms/step |
| 2,048 | 263.44 ms | 255.76 ms | 32.67 ms/step | 32.58 ms/step |
| 8,192 | 1,311.79 ms | 1,236.59 ms | 36.40 ms/step | 34.73 ms/step |

These are medians of six samples per mode. The 8K median difference is 75.2 ms (5.7%); compared with the earlier vLLM first-token median of 903.1 ms, the remaining Xinfer prefill gap is 333 ms (1.37×). Host fuzz processes consumed several CPU cores during this sweep and individual baseline 8K samples ranged from 1,270 to 1,633 ms, so the end-to-end gain is less certain than the exact-logit result. The runs did not remeasure vLLM. Raw Xinfer logs are `/tmp/xinfer-gemma4-layer-scale-v2-{baseline1,fused1,fused2,baseline2}.log` on the measurement host. An initial sweep accidentally disabled the fused final RMSNorm in its baseline and is excluded from these numbers. The 211 ms summed CPU API duration for scalar uploads cannot be counted as recoverable wall time without a paired measurement.

## Resident NVFP4 scales and fused gated GELU quantization

The next opt-in path stores each linear layer's two calibrated NVFP4 hardware scales in a model-owned CUDA tensor (`XINFER_NVFP4_RESIDENT_SCALES=1`). Its attention-rs API accepts that tensor for the hardware FP4 GEMM, avoiding the two four-byte host uploads on each call. Online activation scaling continues to compute its scales on device and takes precedence. Lazy Nemotron expert weights retain the old scratch path, so their strict GPU cache budget is unchanged.

For BF16 NVFP4 Gemma 4, `XINFER_GEMMA4_FUSED_GELU_FP4=1` computes the existing rounded BF16 gated GELU inside hardware FP4 activation quantization for the down projection. The merged gate/up output remains BF16; the intermediate activated BF16 tensor and its separate kernel launch are removed. This path requires CUTLASS on SM100+, checkpoint-calibrated activation scaling, and hardware FP4 decode for single-token calls. Unsupported cases use the existing path. Both options are disabled by default.

The kernel was introduced in the `moriyoshi/attention.rs` fork revision `9a00ef599e16cc2f2d5c43633885eccd8531f18e`. Focused synthetic CUDA tests compare every BF16 output word with materialized gated GELU and ordinary FP4 quantization, both with scratch-uploaded and resident scales. On the real E4B NVFP4 checkpoint, separate resident-scale and gated-quantization parity tests each found maximum absolute logit difference **zero** and top-1 agreement at all 147 positions: 40, 2,048, and 8,192-token prefixes plus 48 teacher-forced continuation positions per prefix.

An isolated GB10 sweep compared baseline, fused GELU/FP4, resident scales, and both in the order baseline/fused/fused/baseline/resident/both/both/resident. Each run loaded the same checkpoint, used the same token files and default Gemma 4 fusions, set `XINFER_GEMMA4_BF16_GEMV=1` and `XINFER_NVFP4_HW_DECODE=1`, and timed three samples per size after a short warmup. The table gives six-sample medians per mode; model loading and compilation are excluded.

| New options | 2K prefill | 8K prefill | 2K decode | 8K decode |
| --- | ---: | ---: | ---: | ---: |
| Neither | 228.42 ms | 1,177.77 ms | 28.62 ms/step | 30.82 ms/step |
| Fused GELU/FP4 | 220.28 ms | 1,084.58 ms | 29.36 ms/step | 31.46 ms/step |
| Resident scales | 231.70 ms | 1,128.66 ms | 28.84 ms/step | 30.86 ms/step |
| Both | 215.24 ms | 1,058.72 ms | 28.30 ms/step | 30.50 ms/step |

At 8K, the combined option saves 119.05 ms (10.1%) against this sweep's baseline; the fused GELU/FP4 option alone saves 93.20 ms (7.9%). Resident scales alone have a smaller and inconsistent effect: the 2K median is 3.28 ms slower. The six 8K combined samples range from 1,052.05 to 1,070.10 ms; the fused-only set includes one 1,765.21 ms outlier, retained in its median and range. All 72 first-token choices were `818`. No other GPU compute context appeared at any run boundary. Raw logs and the parsed `summary.json` are under `/tmp/xinfer-gemma4-nvfp4-modes-20261018/` on the measurement host.

The earlier separate-time vLLM medians were 903.1 ms for 8K first token and 25.4 ms/step for decode. Thus the combined Xinfer result is 1.17× and 1.20× those values respectively. The paired Xinfer sweep supports the 10.1% within-engine gain; it does not remove scheduler, CUDA graph, or host-time differences from the cross-engine comparison. The prior 211 ms summed copy-API duration was not directly recoverable wall time.

An Nsight Systems comparison of one 8K prefill window per mode, on the same Xinfer revision, found 301 CUTLASS FP4 GEMMs in both. Baseline versus both-options kernel sums were 953.1 versus 892.4 ms. Attention was 304.4 versus 301.7 ms and CUTLASS GEMM was 249.7 versus 251.7 ms. The separate gated GELU kernels (89.0 ms) disappeared; 42 fused gated-GELU/FP4 quantization kernels took 69.0 ms. Resident scales reduced host-to-device copy count from 703 to 101, but GPU copy time was only 0.36 versus 0.07 ms. These sums explain some of the wall difference without assigning all of it to a single kernel.

Both traces also show 392 GPU memsets totaling about **114 ms** and 24.1 GB of logical writes inside the prefill window. Forty-two of them each clear a 320 MiB FP4 GEMM output immediately before the native CUTLASS epilogue runs. The traces are `/tmp/xinfer-gemma4-nvfp4-{baseline,both}-profile.nsys-rep` on the measurement host. A guarded experiment attempted to omit the clear for native BF16 CUTLASS, whose epilogue declares `ElementC=void`. The real Gemma 4 checkpoint gave bit-identical logits at 40, 2,048, and 8,192 prefix tokens and 48 continuations per prefix. **The experiment was reverted:** synthetic full-output comparisons found unwritten or differing BF16 words at legal shapes including M=3/N=160/K=128, M=1/N=128/K=128, and M=2048/N=10240/K=2560. CUDA `compute-sanitizer --tool initcheck` also reported uninitialized device-to-host reads in the attempted zero-free output, matching the failing word regions. A follow-up disabled SM120 programmatic dependent launch and matched zero/one/empty output words at those shapes, but initcheck still flagged the empty buffer. The launch change alone measured 1,157.5 → 1,169.5 ms at 8K prefill and was not retained. The zero-initialized output therefore remains necessary until the launch or tile write coverage is corrected and proven across supported shapes. The retained fork revision `4d51b974c3cd35097c2913ef3cac55474903848c` makes native BF16 CUTLASS launch rejection an error rather than silently returning a partially computed tensor; it does not change normal inference latency.

## Opt-in packed QKV for NVFP4 Gemma 4

`XINFER_GEMMA4_PACKED_QKV_NVFP4=1` concatenates the native packed Q, K, and V weights and E4M3 block scales into one projection for Gemma 4 layers that produce all three. The path is limited to BF16 CUDA CUTLASS on SM100+, with no QKV bias or gated query. It packs only when the three checkpoint input and global calibration scalars match bit for bit and their local dimensions align; otherwise it uses the separate projections. All 24 QKV-producing layers in the measured E4B checkpoint satisfy those checks. The other 18 layers reuse earlier K/V and keep their query-only projection.

The merged output is reshaped into heads before slicing, preserving its token-row stride. The fused Q/K norm and RoPE kernel reads those strided views directly. V is copied into a contiguous tensor because the pinned attention-rs KV writer does not accept a V row stride. The single-token no-cache shortcut retains contiguous splitting. The ordinary path is unchanged when the option is unset.

The real checkpoint produced **bit-identical logits** against separate QKV for 40, 2,048, and 8,192-token prefixes and 48 teacher-forced continuation positions after each prefix: maximum absolute difference zero, top-1 agreement 147/147. A CUDA regression also compares every BF16 output word of the Q/K kernel with contiguous versus packed-row-strided inputs. A CPU regression rejects packing when calibration scalars differ.

An isolated GB10 baseline/packed/packed/baseline sweep used the same checkpoint, token IDs, binary, and existing resident-scale and fused-GELU/FP4 options. Each mode has six samples per size; model loading and compilation are excluded.

| Prompt tokens | Separate prefill | Packed prefill | Separate decode | Packed decode |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 33.55 ms | 33.12 ms | 29.74 ms/step | 28.48 ms/step |
| 2,048 | 242.90 ms | 244.00 ms | 32.38 ms/step | 31.09 ms/step |
| 8,192 | 1,192.98 ms | 1,179.64 ms | 34.73 ms/step | 33.42 ms/step |

The 8K prefill gain is 13.34 ms (1.1%); the 2K prefill result is effectively flat. The 8K decode gain is 1.31 ms/step (3.8%). These runs did not remeasure vLLM, and other host work made their absolute times higher than the earlier isolated Xinfer sweep, so the table supports only the within-run comparison. The Xinfer-vLLM gap remains substantial.

An Nsight comparison before the strided-view change showed why an ordinary packed projection was insufficient. Per 8K prefill window, FP4 activation quantization fell from about 139 to 127 ms, while contiguous Q/K/V splitting added 72 BF16 copy kernels taking about 23 ms. The strided-view path removes the Q/K copies; V remains contiguous for KV correctness. Raw sweep logs are under `/tmp/xinfer-gemma4-qkv-bench-20261019/`, and the initial packed/split Nsight exports are `/tmp/xinfer-gemma4-qkv-{baseline,packed}-profile.sqlite` on the measurement host. This option adds temporary concatenation buffers during model construction; peak construction memory was not measured.

## Opt-in 512-wide attention shared-memory layout

`XINFER_FLASH_512_SWIZZLED=1` selects a shared-memory permutation for native
512-wide FP16/BF16 paged prefill. The attention-rs implementation is pinned at
`f011ee482357a62774f38d6acffab2fc0d14938b`. It XORs aligned 16-byte chunks
within each 128-byte row segment, avoiding the original power-of-two row-stride
bank conflicts without increasing the 101,120-byte shared-memory arena. Global
Q/K/V layouts, logical positions, masks, softmax and MMA accumulation order stay
the same. The default, decode, FP8 KV and other head widths retain their existing
paths. Runtime validation so far is limited to GB10; this remains opt-in.

In the retained 8K trace, the seven full-attention layers accounted for about
256 ms of 317 ms summed attention time. An isolated BF16 kernel experiment
reduced one 8K full-attention launch from approximately 35 to 21 ms, with all
33,554,432 output words identical. Hardware counter access was denied by the
host, so bank-conflict counts were not measured directly.

A 2026-10-22 baseline/swizzled/swizzled/baseline sweep used the same binary, checkpoint,
token streams and BF16 KV, with resident scales, fused GELU/FP4, packed QKV and
BF16 GEMV enabled in both modes. Medians use six samples per size and mode;
loading and compilation are excluded. No other compute context was visible
before or after any run; minimum monitored `MemAvailable` was 101.0 GiB.

| Prompt tokens | Original prefill | Swizzled prefill | Original decode | Swizzled decode |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 27.82 ms | 28.17 ms | 24.67 ms/step | 24.77 ms/step |
| 2,048 | 216.06 ms | 210.28 ms | 27.19 ms/step | 27.27 ms/step |
| 8,192 | 1,075.49 ms | 974.22 ms | 29.44 ms/step | 29.46 ms/step |

The 8K prefill reduction is 101.28 ms (9.4%); 2K improves by 2.7%. Decode does
not use this kernel and is effectively unchanged. A fresh vLLM control was
initially deferred when another GPU job started. The older vLLM 8K result of
903.1 ms would put this Xinfer result about 1.08x slower, but that is a comparison
between separate measurement windows, not a new paired engine comparison.

`tests/flash_512_swizzle.rs` compares every FP16/BF16 output word, including
partial tiles, shuffled physical pages, unequal batched sequence lengths,
continued prefill, sliding windows and softcapping. A separate FP64 CPU softmax
oracle has maximum absolute differences 0.00266 (BF16) and 0.000340 (FP16).
CUDA initcheck reports zero errors with an uninitialized output allocation.
`real_e4b_nvfp4_swizzled_attention_matches_baseline` checks the real checkpoint
at 40/2K/8K, both full and split prefill, followed by 48 teacher-forced positions
per schedule: maximum logit difference zero, top-1 agreement 297/297.

```sh
cargo test --release --features cuda,cutlass --test flash_512_swizzle \
  swizzled_512_matches_original_and_cpu -- --exact --ignored --nocapture
XINFER_GEMMA4_NVFP4_CHECKPOINT="$MODEL" XINFER_GEMMA4_BENCH_DIR="$BENCH" \
XINFER_GEMMA4_BF16_GEMV=1 XINFER_NVFP4_HW_DECODE=1 \
XINFER_NVFP4_RESIDENT_SCALES=1 XINFER_GEMMA4_FUSED_GELU_FP4=1 \
XINFER_GEMMA4_PACKED_QKV_NVFP4=1 \
cargo test --release --features cuda,cutlass --test gemma4_shared_kv \
  real_e4b_nvfp4_swizzled_attention_matches_baseline -- --exact --ignored --nocapture
```

For the existing latency test, set `XINFER_FLASH_512_SWIZZLED` to `0` and `1`
in interleaved process runs, keeping all other options identical. Raw results,
watchdog records and the kernel experiment are retained under
`/tmp/xinfer-gemma4-smem512-experiment/` on the measurement host.

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
