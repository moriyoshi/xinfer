// Nemotron-H BF16 paged prefill, specialized for head width 128 on GB10.
// The tiled core and compatibility helpers retain their Apache-2.0 notices.
#define FLASH_HDIM 128
#include "flash_prefill_paged.cuh"

extern "C" int xinfer_nemotron_flash_prefill_bf16(
    const void* q, const void* k_cache, const void* v_cache,
    const void* k_direct, const void* v_direct, bool direct_current, void* output,
    const int* block_tables, unsigned int block_table_stride,
    const unsigned int* cu_seqlens_q, const unsigned int* context_lens,
    unsigned int num_seqs, unsigned int max_seqlen_q,
    unsigned int num_q_heads, unsigned int num_kv_heads,
    unsigned int cache_block_size, float scale, bool grouped_heads,
    long long stream_raw) {
  using Half = __nv_bfloat16;
  constexpr int shared_bytes =
      (BR * HDIM_PAD + 2 * BC * HDIM_PAD + BC * HDIM_PAD +
       BR * (BC + PAD_P)) * sizeof(Half);
  if (!num_kv_heads || (grouped_heads && (num_q_heads % 4 != 0 ||
      (num_q_heads / num_kv_heads) % 4 != 0)))
    return static_cast<int>(cudaErrorInvalidValue);
  auto kernel = grouped_heads ? flash_prefill_paged<Half, true>
                              : flash_prefill_paged<Half, false>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (err != cudaSuccess) return static_cast<int>(err);
  const unsigned int heads_per_block = grouped_heads ? 4 : 1;
  const unsigned int rows_per_head = grouped_heads ? BR / 4 : BR;
  dim3 grid(num_q_heads / heads_per_block,
            (max_seqlen_q + rows_per_head - 1) / rows_per_head, num_seqs);
  kernel<<<grid, NUM_THREADS, shared_bytes,
           reinterpret_cast<cudaStream_t>(stream_raw)>>>(
      static_cast<const Half*>(q), static_cast<const Half*>(k_cache),
      static_cast<const Half*>(v_cache), static_cast<const Half*>(k_direct),
      static_cast<const Half*>(v_direct), direct_current, static_cast<Half*>(output),
      block_tables, block_table_stride, cu_seqlens_q, context_lens,
      num_q_heads, num_kv_heads, FLASH_HDIM, cache_block_size,
      0, 1, scale, 0.0f);
  return static_cast<int>(cudaPeekAtLastError());
}
