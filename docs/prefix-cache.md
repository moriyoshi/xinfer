# Prefix Cache (KV Reuse)

Prefix cache lets xInfer reuse KV cache blocks from prior requests when a new
prompt shares a prefix. This accelerates consecutive requests with overlapping
history (for example, chat sessions that replay the same system + earlier turns).

## How it works
- Finished sequences contribute full KV blocks to a global prefix cache.
- New requests find the longest cached prefix (block-aligned) and reuse those blocks.
- Remaining tokens are prefetched as usual, with KV writes continuing after the cached prefix.

Prefix cache is block-granular: only full KV blocks are reused. If the common
prefix ends mid-block, the tail of that block is recomputed. When a prompt is
fully cached at block boundaries, the last block is recomputed to ensure a
non-empty prefill step for correct sampling.

## Flags
- Prefix cache is **enabled by default**. Use `--disable-prefix-cache` to turn it off.
- `--prefix-cache-max-tokens <N>`: cap cache size in tokens (rounded down to block size).

If `--prefix-cache-max-tokens` is not set, defaults are:
- Normal mode: ~50% of GPU KV blocks
- PD server: ~75% of GPU KV blocks
- PD client: ~35% of GPU KV blocks

## Hybrid Mamba Snapshot Stride
For hybrid Mamba models (for example Qwen3.5), prefix reuse also needs a
compatible Mamba snapshot at the matched boundary.

Use environment variable `XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS` to control
sparse snapshot capture during decode (larger stride side usefull for limited GPU memory):
- Default: `1` blocks
- Minimum valid value: `1` (capture every block)
- Effective snapshot boundary in tokens: `block_size * stride`

Example with default `block_size=64` and stride `8`:
- Decode snapshot boundary is every `512` tokens.
- Effective hybrid prefix reuse is aligned to the nearest captured boundary.

This setting only sparsifies decode-time snapshot capture. Prompt/prefill
snapshot capture remains dense.

## Notes
- Prefix cache uses the same KV memory pool as active sequences. A larger cache
  reduces the maximum number of concurrent tokens available for new requests.
- Cached KV reuse is automatic; no `session_id` is required.
- Sliding window attention limits how much cached context is effectively used.

## CPU swap with prefix cache

When prefix cache is enabled, live sequence preemption is partial:

- Leading prefix/shared blocks stay on GPU and keep their sequence reference.
- Only the sequence-owned suffix blocks are copied to CPU swap.
- Swap-in reuses the retained GPU prefix blocks, allocates new suffix blocks,
  and copies only the suffix KV back from CPU.

This is different from prefix-cache offload. Prefix-cache offload only helps when
an evicted cached prefix is requested again later. Partial sequence swap helps
immediately under memory pressure because it avoids recomputing the active
sequence suffix after preemption, even when no future request reuses that suffix.
For hybrid Mamba/GDN models, the active recurrent state remains tied to the
swapped sequence lifecycle and is not released until the sequence finishes or is
cancelled. On CUDA, Mamba/GDN prefix snapshots use a separate two-tier cache:
device snapshots are the fast path, and snapshots evicted from device memory are
spilled to CPU so a later KV prefix hit can promote the matching recurrent state
back instead of falling back to a shorter partial hit. Metal uses unified memory,
so this CPU spill tier is disabled there. The CUDA CPU snapshot tier is 4x the
device snapshot capacity. Prompt/prefill snapshots and final decode-boundary
snapshots are protected from ordinary decode-time snapshot churn. When the CPU
snapshot tier is full, LRU eviction frees at least 10% of the tier in one batch
before accepting new GDN/Mamba snapshot offloads.

## Portable GDN state for separate-GPU prefill

`Qwen3_5ForCausalLM::export_gdn_state` and its MoE counterpart export all Gated
DeltaNet convolution and recurrent states for one sequence after a completed
prefill boundary. They return a `GdnStateSnapshot` with version, token count,
absolute decoder layer order, FP32 shapes and dtypes, tensor-parallel rank and
world size, a payload
SHA-256, and exact little-endian FP32 bits. `to_bytes` and `from_bytes` provide
a portable binary envelope. `import_gdn_state` validates the complete snapshot
before allocating an unused sequence slot on the target model.

```rust
let snapshot = prefill_model.export_gdn_state(seq_id, prefix_tokens, fingerprint)?;
let bytes = snapshot.to_bytes()?;
let snapshot = GdnStateSnapshot::from_bytes(&bytes)?;
decode_model.import_gdn_state(new_seq_id, prefix_tokens, fingerprint, &snapshot)?;
```

When the caller stores or transports the complete envelope, use
`export_gdn_state_bytes` and `import_gdn_state_bytes` on the dense or MoE model.
They preserve the v1 bytes and check the payload once on each side. The
snapshot-object methods continue to validate their public mutable fields.
Qwen3-VL forwards these byte APIs when its text model is a Qwen3.5 hybrid;
its fingerprint must cover both vision and text weights. Qwen4 exposes the
same GDN byte APIs when PLE is absent. Qwen4 with PLE rejects export and import
because its PLE convolution and token-context state is additional recurrent
state that this GDN envelope does not contain.

The caller supplies the same 32-byte SHA-256 model fingerprint on both sides.
It must identify weights, adapters, and numerical execution settings; xinfer
does not derive it from loaded weights. The caller must also restore attention
KV from the **same exact token boundary** before decoding, and must give the
imported sequence an unused ID with enough Mamba cache capacity. An existing
slot is rejected so an active sequence cannot be overwritten. The model API
does not itself store, transport, or publish a combined KV and GDN bundle.

## Portable Nemotron-H Mamba state

`NemotronHForCausalLM::export_mamba_state` exports the FP32 convolution and
SSM state for every Mamba layer at a completed token boundary. The returned
`NemotronMambaSnapshot` carries a version, absolute layer order, tensor shapes
and dtypes, tensor parallel layout, token boundary, caller supplied model
fingerprint, and payload SHA-256. `to_bytes` and `from_bytes` use a portable
binary envelope with little-endian FP32 payload bits.

```rust
let snapshot = prefill_model.export_mamba_state(seq_id, prefix_tokens, fingerprint)?;
let bytes = snapshot.to_bytes()?;
let snapshot = NemotronMambaSnapshot::from_bytes(&bytes)?;
decode_model.import_mamba_state(new_seq_id, prefix_tokens, fingerprint, &snapshot)?;
```

For a persisted envelope, `export_mamba_state_bytes` and
`import_mamba_state_bytes` avoid repeated payload checksum passes while keeping
the same v1 format. The bytes-first import keeps its validated snapshot private
until state installation; the snapshot-object API still validates on import.

The 32-byte fingerprint must identify compatible weights, adapters, and
numerical execution settings. xinfer cannot derive it from loaded weights.
Restore attention KV from the same exact token boundary before continuation;
the Mamba snapshot contains no attention KV. Imports reject incompatible
layouts, boundaries, fingerprints, corrupt payloads, occupied sequence IDs,
and exhausted state capacity. The pinned Japanese 9B model has 27 Mamba
layers and exports 145,539,072 payload bytes per sequence. The model API does
not assemble or persist the combined Mamba and attention bundle.

## Inspecting cache hits

Chat completion responses include the prefix-cache hit count under
`usage.prompt_tokens_details.cached_tokens` (OpenAI extension). The field is
omitted when no hits occurred, so existing single-turn responses keep their
shape:

```json
"usage": {
  "prompt_tokens": 499,
  "completion_tokens": 16,
  "total_tokens": 515,
  "prompt_tokens_details": { "cached_tokens": 480 }
}
```

In Python (offline batch), call `engine.get_num_cached_tokens_for_seq(seq_id)`
on the `seq_id` returned in each `GenerationOutput`.

For models that emit `<think>…</think>` reasoning blocks, responses also
include `usage.completion_tokens_details.reasoning_tokens` so clients can
attribute completion cost across reasoning vs final-answer output:

```json
"usage": {
  "prompt_tokens": 12,
  "completion_tokens": 256,
  "total_tokens": 268,
  "completion_tokens_details": { "reasoning_tokens": 192 }
}
```
