# Dense packed KV page restore

Xinfer exposes `DensePackedKvPage` and `restore_dense_packed_kv_pages` in
`utils::packed_kv_restore`, plus a `ModelRunner` forwarding method. This path
expands verified 2- or 4-bit pages into allocated BF16 Flash K/V slots. It is
separate from TurboQuant, MLA, DeepSeek V4, and the generic yesno bitplane
codec. It does not change attention kernels or select a precision policy.

## Version 1 page contract

The portable envelope is the eight-byte magic `XPKV\0\0\0\x01` followed by
bincode fixed-width little-endian fields in the order of `DensePackedKvPage`.
`version` is 1. The SHA-256 covers a fixed domain tag, version, axis, bits,
geometry, vector lengths, and every payload field in little-endian order.
`from_bytes` rejects trailing bytes, invalid lengths, nonzero code padding,
invalid scales, duplicate/out-of-range exceptions, and checksum changes.

- `axis`: K groups along old tokens; V groups along channels.
- `bits`: 2 or 4. Codes are densely LSB-first packed. K code order is
  `[head, channel, old_token]`; V code order is `[old_token, head, channel]`.
- `tokens`, `heads`, `channels`: one page's output shape `[T,H,D]`.
- `group_size`: maximum elements per affine group along the named axis.
  K parameter order is `[head, channel, token_group]`; V parameter order is
  `[old_token, head, channel_group]`. Each group has interleaved f32
  `(minimum, step)`. Decode uses f64 `minimum + code * step`, then f32 and
  round-to-nearest-even BF16, matching the independent compact decoder.
- `tail_tokens`: final tokens copied exactly from `tail_bf16`. K tail order is
  `[head, channel, tail_token]`; V tail order is `[tail_token, head, channel]`.
  A page can be entirely exact with empty codes and parameters.
- `exceptions`: strictly increasing token-major output indexes before the
  tail, each with exact BF16 bits. Exceptions override affine reconstruction.

The page envelope validates content and geometry. The peer bundle must bind
it to the model weights, prefix boundary, logical token address, and attention
layer. The restore request supplies the physical block and token offset; its
layer index is the **compact index among attention KV pairs**, which can differ
from a model's absolute layer number. Shifou must verify that mapping before
calling xinfer. Xinfer rejects overlapping requests, incompatible dimensions,
noncontiguous/non-BF16 destinations, and unsupported cache backends before
writing any slot. `ModelRunner` also rejects TurboQuant mode; a direct caller
must provide standard Flash KV tensors rather than its dummy slots. The
caller allocates blocks and installs the attention page table. Restores
complete on the CUDA stream before returning.

The GPU path combines up to 128 pages into one set of packed uploads and one
kernel launch, stopping before the batch exceeds 64 MiB of payload unless a
single page is larger. K/V pages, bit widths, tail lengths, and exception
counts may
vary within a batch. Pages are bucketed by output size so a large page does
not create mostly idle blocks for tiny pages. The first call compiles the
kernel with NVRTC; later calls reuse it. The CPU oracle `decode_cpu_bf16` and
checked-in independent shifou compact fixtures verify byte-for-byte output
for K/V at 2/4 bits.

## Integration and quality gate

Shifou's adapter must transcode its current variable-width yesno bitplanes,
f64 group parameters, and exceptions to this dense contract, or encode pages
directly in this format. It must retain exact recent BF16 tokens, verify its
outer model/session address, and pass only selected, nonoverlapping pages. The
current shifou adaptive selector has not shown reliable held-out quality, so
this decoder alone is not a serving-quality claim. Generic f64 affine
parameters cannot always be represented by the f32 parameters here; exact
transcoding may require re-encoding or exceptions and can increase size.

On one GB10, a warmed debug-build xinfer call for the 7,000-token, 8-head,
128-channel tile took p50 1.07/1.89 ms for K 2/4-bit and 2.62/3.43 ms for V
2/4-bit. This includes envelope validation, host staging, GPU allocation and
uploads, expansion, and synchronization; it excludes peer reads, page-table
installation, and attention. The standalone shifou CUDA kernel benchmark
reported 0.23–0.27 ms but excluded several of these costs. Neither is an
end-to-end serving result.

An end-to-end gate should restore a complete prefix bundle into a second model
instance, compare every restored BF16 slot against shifou's CPU decoder, then
compare continuation logits and greedy tokens against the same BF16-restored
page policy. Evaluate held-out KL, ranking, and task quality against an exact
BF16 prefix separately from decoder parity. Include storage read, validation,
page-table installation, GPU expansion, and continuation in latency results.
