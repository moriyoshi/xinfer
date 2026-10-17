# Gemma 4 E4B native NVFP4 loading

Xinfer can load the text path of a ModelOpt NVFP4 Gemma 4 E4B checkpoint. The ordinary projections already use the quantized linear loader; the per-layer model projection had a separate BF16-only loader and interpreted packed `uint8` weights as dense values. It now uses the same native NVFP4 loader when the projection has ModelOpt's block and global scales. The existing dense path remains for BF16 checkpoints. Direct `Gemma4ForCausalLM` calls also set the prefill/decode linear mode from their input metadata.

The real-checkpoint test used `Neural-ICE/Gemma-4-E4B-it-NVFP4` revision `cccd4fa40a99a8d525e050f014ea21dc2b680995` against `google/gemma-4-E4B-it` BF16 revision `ee0ef6023621cff504d758262d4e04895a5af4a2` on a GB10. The two safetensors shards of the NVFP4 checkpoint matched their published LFS SHA-256 hashes. Weight files total 10,202,718,788 bytes versus 15,992,595,884 BF16 bytes, 36.2% less. The published NVFP4 conversion is a third-party checkpoint; this test did not regenerate it from the pinned BF16 revision, so the paired results measure their combined difference. Both models use the same text configuration and vocabulary.

The test uses BF16 attention KV for both models to isolate weight quantization. It feeds a 40-token chat prompt, compares the prefill result and 48 teacher-forced continuation positions, checks repeatability on the same schedule, and generates 48 greedy tokens with separate caches. `XINFER_GEMMA4_BF16_GEMV=1` was enabled for the BF16 baseline. The measured decode times are within paired runs on a shared host, not an isolated performance guarantee.

| NVFP4 decode mode | Prefix top token ID, NVFP4/BF16 | Teacher-forced top choices | Largest absolute logit difference | NVFP4 ms/token | BF16 ms/token | Greedy matching prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware FP4, default | 818/818 | 45/48 | 22.680 | 31.579 | 48.484 | 37/48 |
| Software W4A16 (`XINFER_NVFP4_HW_DECODE=0`) | 818/818 | 46/48 | 16.895 | 26.170 | 49.231 | 5/48 |

The prefix's maximum absolute logit difference was 14.174 in both modes. Repeated prefill logits and the first four repeated decode logits were bit-identical within each NVFP4 mode. The generated responses in this one sky-color prompt were coherent in both modes. Quantization changed some top choices, and free-running branches can diverge after a choice changes. These results do not establish quality on llmbda's 80-task workload or on image/audio inputs. Keep BF16 KV when comparing weight formats; the published checkpoint advertises FP8 KV, which is a separate accuracy variable.

To run the opt-in regression:

```sh
XINFER_GEMMA4_NVFP4_CHECKPOINT=/path/to/Neural-ICE-Gemma-4-E4B-it-NVFP4 \
XINFER_GEMMA4_CHECKPOINT=/path/to/google-gemma-4-E4B-it \
XINFER_GEMMA4_REFERENCE_DIR=/path/to/gemma4/ref \
XINFER_GEMMA4_BF16_GEMV=1 \
cargo test --release --features cuda,cutlass --test gemma4_shared_kv \
  real_e4b_nvfp4_load_prefill_and_continuation -- --nocapture
```

The reference directory is described in [the shared-KV test notes](gemma4-e4b-shared-kv.md). The test requires a CUDA GPU and the two real checkpoints; it skips unless `XINFER_GEMMA4_NVFP4_CHECKPOINT` is set. The software comparison additionally sets `XINFER_NVFP4_HW_DECODE=0`, a process-wide mode affecting other NVFP4 models. The default hardware mode remains unchanged.

For a new conversion, NVIDIA's [Model Optimizer HF PTQ example](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md) documents `nvfp4` and the more conservative `nvfp4_mlp_only` formats, with `--kv_cache_qformat none` for unquantized KV. Conversion accuracy and Gemma 4 multimodal behavior require separate validation on the resulting checkpoint.
