# Gemma 4 E4B shared K/V validation

Tested on a GB10 with `google/gemma-4-E4B-it` BF16 weights at Hugging Face revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. The reference was `transformers` 5.6.0 in `nvcr.io/nvidia/vllm:26.05-py3`, using the model's chat template. The Xinfer test is `tests/gemma4_shared_kv.rs`; set `XINFER_GEMMA4_CHECKPOINT` and `XINFER_GEMMA4_REFERENCE_DIR` to enable the real-checkpoint differential. The reference directory contains `tokens.json`, `hf.f32`, and `hf.json` from the 40-token prompt; optional `hf_long.f32` contains last-token logits for the same tokens repeated to length 640.

## Cache contract

For E4B, `num_hidden_layers=42` and `num_kv_shared_layers=18`. Layers 0–23 write their own K/V. Shared sliding-attention layers 24–41 read layer 22's K/V; shared full-attention layers read layer 23's K/V. Shared layers project only Q and never write K/V. `Gemma4ForCausalLM::kv_cache_source_layers()` returns the exact per-layer mapping, and `num_kv_cache_layers()` returns the compact pair count.

`Gemma4ForCausalLM::forward` accepts either the legacy 42-pair cache or a compact 24-pair cache. In the legacy layout, pairs 24–41 are ignored. Export and restore only pairs 0–23 for compact resume, retaining each pair's original layer index and head shape. The general engine allocator still constructs 42 pairs; direct callers can use 24 pairs. At BF16, E4B's logical K/V payload is 100,352 bytes/token with 42 pairs and 57,344 bytes/token with 24 pairs.

## Differential results

| Check | Result |
| --- | --- |
| 40-token next-token choice | Hugging Face 818 (`The`); Xinfer 818. Previous Xinfer was 902. |
| 40-token last-position logits | Maximum absolute difference 0.6853; previous Xinfer was 37.7. |
| 48-token greedy continuation | 29/48 token IDs match; first difference at token 14. Xinfer text remains a coherent explanation of Rayleigh scattering and names Lord Rayleigh. This is numerical agreement, not bit parity with Hugging Face. |
| 40-token full vs 32+8 continued prefill | Same top choice; maximum absolute difference 0.6537. Repeating the same continued schedule is bit-identical. |
| 40-token full vs 32+8 single-token decode | Same top choice; maximum absolute difference 0.5626. Repeating the same decode schedule is bit-identical. |
| 42-pair vs 24-pair cache | Bit-identical logits for the full prefill. |
| 640-token full vs 520+120 continued prefill | Same top choice; maximum absolute difference 0.7941, crossing the 512-token sliding window. |
| 640-token full vs Hugging Face | Both choose token 818; maximum absolute difference 1.9344. |

The remaining BF16 differences grow across layers and longer contexts; they can change later greedy choices. The input token embedding matched the reference exactly in a layer trace, while the first layer already differed slightly after the per-layer embedding and attention computation. This change corrects the missing shared-K/V behavior, but does not claim exact numerical equivalence to PyTorch. Shared attention currently requires the CUDA flash path and native BF16 K/V cache dtype. The ordinary engine retains the legacy cache allocation until its allocator is taught the compact mapping.
