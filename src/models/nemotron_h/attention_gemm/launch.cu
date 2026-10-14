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
    unsigned int cache_block_size, float scale, long long stream_raw) {
  using Half = __nv_bfloat16;
  constexpr int shared_bytes =
      (BR * HDIM_PAD + 2 * BC * HDIM_PAD + BC * HDIM_PAD +
       BR * (BC + PAD_P)) * sizeof(Half) + BR * 2 * sizeof(float);
  auto kernel = flash_prefill_paged<Half>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (err != cudaSuccess) return static_cast<int>(err);
  dim3 grid(num_q_heads, (max_seqlen_q + BR - 1) / BR, num_seqs);
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
