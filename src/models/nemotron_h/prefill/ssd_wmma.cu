// Experimental BF16 tensor-core SSD output for the 64x64x128 Nemotron-3
// Mamba mixer. The FP32 chunk summaries and carried-state prefix remain in
// scan.cu; this kernel only replaces the second recurrent output pass.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;
using bf16 = __nv_bfloat16;
constexpr int CT = 128;
constexpr int TILE = 16;
constexpr int HD = 64;
constexpr int SD = 128;

__global__ void cb_gemm(const float* conv, bf16* cb, int tokens, int heads,
                        int groups, int conv_dim) {
  const int group = blockIdx.x;
  const int chunk = blockIdx.y;
  const int tm = blockIdx.z / 8;
  const int tn = blockIdx.z % 8;
  const int start = chunk * CT;
  const int inner = heads * HD;
  __shared__ bf16 a[TILE * TILE], b[TILE * TILE];
  __shared__ float result[TILE * TILE];
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int sk = 0; sk < SD; sk += TILE) {
    for (int j = threadIdx.x; j < TILE * TILE; j += blockDim.x) {
      const int r = j / TILE, c = j % TILE;
      const int tr = start + tm * TILE + r;
      const int tc = start + tn * TILE + c;
      a[j] = __float2bfloat16_rn(tr < tokens
          ? conv[tr * conv_dim + inner + groups * SD + group * SD + sk + c]
          : 0.0f);
      b[j] = __float2bfloat16_rn(tc < tokens
          ? conv[tc * conv_dim + inner + group * SD + sk + r]
          : 0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16, wmma::row_major> bf;
    wmma::load_matrix_sync(af, a, TILE);
    wmma::load_matrix_sync(bf, b, TILE);
    wmma::mma_sync(acc, af, bf, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(result, acc, TILE, wmma::mem_row_major);
  __syncwarp();
  for (int j = threadIdx.x; j < TILE * TILE; j += blockDim.x) {
    cb[(((chunk * groups + group) * CT + tm * TILE + j / TILE) * CT)
       + tn * TILE + j % TILE] = __float2bfloat16_rn(result[j]);
  }
}

// Reuse each C and B tile across a 4x4 warp array. The WMMA instruction and
// eight state-reduction steps are identical to cb_gemm; only the block's
// output footprint changes from 16x16 to 64x64.
template<int REDUCTION_TILE>
__global__ void cb_gemm64(const float* conv, bf16* cb, int tokens,
                          int heads, int groups, int conv_dim) {
  const int group = blockIdx.x;
  const int chunk = blockIdx.y;
  // Only the lower triangle is consumed by ssd_output (i <= t). The
  // upper-right 64x64 tile is entirely future-token products, so do not
  // launch a block for it. z maps to (0,0), (1,0), (1,1).
  const int tile_m = blockIdx.z == 0 ? 0 : 1;
  const int tile_n = blockIdx.z == 2 ? 1 : 0;
  const int warp = threadIdx.x / 32;
  const int warp_m = warp / 4;
  const int warp_n = warp % 4;
  const int start = chunk * CT;
  const int inner = heads * HD;
  constexpr int STAGING_BYTES = 2 * 64 * REDUCTION_TILE * sizeof(bf16);
  constexpr int RESULT_BYTES = 64 * 64 * sizeof(float);
  constexpr int SHARED_BYTES = STAGING_BYTES > RESULT_BYTES
      ? STAGING_BYTES : RESULT_BYTES;
  __shared__ __align__(16) unsigned char scratch[SHARED_BYTES];
  bf16* a = reinterpret_cast<bf16*>(scratch);
  bf16* b = a + 64 * REDUCTION_TILE;
  float* result = reinterpret_cast<float*>(scratch);
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int sk = 0; sk < SD; sk += REDUCTION_TILE) {
    for (int j = threadIdx.x; j < 64 * REDUCTION_TILE; j += blockDim.x) {
      const int r = j / REDUCTION_TILE, c = j % REDUCTION_TILE;
      const int t = start + tile_m * 64 + r;
      a[j] = __float2bfloat16_rn(t < tokens
          ? conv[t * conv_dim + inner + groups * SD + group * SD + sk + c]
          : 0.0f);
    }
    for (int j = threadIdx.x; j < REDUCTION_TILE * 64; j += blockDim.x) {
      const int r = j / 64, c = j % 64;
      const int t = start + tile_n * 64 + c;
      b[j] = __float2bfloat16_rn(t < tokens
          ? conv[t * conv_dim + inner + group * SD + sk + r]
          : 0.0f);
    }
    __syncthreads();
    for (int sub = 0; sub < REDUCTION_TILE; sub += TILE) {
      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16,
                     wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16,
                     wmma::row_major> bf;
      wmma::load_matrix_sync(
          af, a + warp_m * TILE * REDUCTION_TILE + sub, REDUCTION_TILE);
      wmma::load_matrix_sync(bf, b + sub * 64 + warp_n * TILE, 64);
      wmma::mma_sync(acc, af, bf, acc);
    }
    __syncthreads();
  }
  // The loop's final barrier makes the staging tiles dead before the result
  // overwrites the same shared storage.
  wmma::store_matrix_sync(result + warp_m * TILE * 64 + warp_n * TILE,
                          acc, 64, wmma::mem_row_major);
  __syncthreads();
  for (int j = threadIdx.x; j < 64 * 64; j += blockDim.x) {
    cb[(((chunk * groups + group) * CT + tile_m * 64 + j / 64) * CT)
       + tile_n * 64 + j % 64] = __float2bfloat16_rn(result[j]);
  }
}

__global__ void decay_prefix(const float* dt_decay, float* log_decay,
                             float* reset_at, int tokens, int heads) {
  const int head = blockIdx.x, chunk = blockIdx.y;
  if (threadIdx.x != 0) return;
  float sum = 0.0f;
  int reset = -1;
  for (int t = chunk * CT; t < min((chunk + 1) * CT, tokens); ++t) {
    const float decay = dt_decay[(t * heads + head) * 2 + 1];
    if (decay == 0.0f) {
      sum = 0.0f;
      reset = t - chunk * CT;
    } else {
      sum += logf(decay);
    }
    log_decay[t * heads + head] = sum;
    reset_at[t * heads + head] = float(reset);
  }
}

__global__ void transpose_starts(const float* starts, bf16* transposed,
                                 int chunks, int heads) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = chunks * heads * SD * HD;
  if (index >= total) return;
  const int dim = index % HD;
  const int state = (index / HD) % SD;
  const int head_chunk = index / (SD * HD);
  transposed[index] = __float2bfloat16_rn(
      starts[(head_chunk * HD + dim) * SD + state]);
}

template<int REDUCTION_TILE, bool DIRECT_STARTS, int OUT_ROWS>
__global__ void ssd_output(const float* conv, const float* dt_decay,
                           const float* log_decay, const float* reset_at,
                           const bf16* cb,
                           const bf16* starts, const float* starts_fp32,
                           const float* d, float* y,
                           int tokens, int heads, int groups, int conv_dim) {
  const int head = blockIdx.x, chunk = blockIdx.y, tile = blockIdx.z;
  const int warp = threadIdx.x / 32;
  const int warp_row = warp / 4, warp_col = warp % 4;
  const int group = head / (heads / groups);
  const int chunk_start = chunk * CT;
  const int inner = heads * HD;
  constexpr int STAGING_BYTES =
      (OUT_ROWS * REDUCTION_TILE + REDUCTION_TILE * HD) * sizeof(bf16);
  constexpr int RESULT_BYTES = OUT_ROWS * HD * sizeof(float);
  constexpr int SHARED_BYTES = STAGING_BYTES > RESULT_BYTES
      ? STAGING_BYTES : RESULT_BYTES;
  __shared__ __align__(16) unsigned char scratch[SHARED_BYTES];
  bf16* lhs = reinterpret_cast<bf16*>(scratch);
  bf16* rhs = lhs + OUT_ROWS * REDUCTION_TILE;
  float* result = reinterpret_cast<float*>(scratch);
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> main_acc;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> state_acc;
  wmma::fill_fragment(main_acc, 0.0f);
  wmma::fill_fragment(state_acc, 0.0f);

  for (int k = 0; k < CT; k += REDUCTION_TILE) {
    for (int j = threadIdx.x; j < OUT_ROWS * REDUCTION_TILE;
         j += blockDim.x) {
      const int row = j / REDUCTION_TILE, col = j % REDUCTION_TILE;
      const int t = chunk_start + tile * OUT_ROWS + row;
      const int i = chunk_start + k + col;
      float value = 0.0f;
      if (t < tokens && i < tokens && i <= t
          && i - chunk_start >= int(reset_at[t * heads + head])) {
        const float dot = __bfloat162float(
            cb[(((chunk * groups + group) * CT + tile * OUT_ROWS + row) * CT)
               + k + col]);
        const float dt = dt_decay[(i * heads + head) * 2];
        const float lag = log_decay[t * heads + head]
                        - log_decay[i * heads + head];
        value = dot * dt * expf(lag);
      }
      lhs[j] = __float2bfloat16_rn(value);
    }
    for (int j = threadIdx.x; j < REDUCTION_TILE * HD;
         j += blockDim.x) {
      const int i = chunk_start + k + j / HD;
      rhs[j] = __float2bfloat16_rn(i < tokens
          ? conv[i * conv_dim + head * HD + j % HD] : 0.0f);
    }
    __syncthreads();
    for (int sub = 0; sub < REDUCTION_TILE; sub += TILE) {
      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16,
                     wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16,
                     wmma::row_major> bf;
      wmma::load_matrix_sync(
          af, lhs + warp_row * TILE * REDUCTION_TILE + sub,
          REDUCTION_TILE);
      wmma::load_matrix_sync(bf, rhs + sub * HD + warp_col * TILE, HD);
      wmma::mma_sync(main_acc, af, bf, main_acc);
    }
    __syncthreads();
  }
  for (int k = 0; k < SD; k += REDUCTION_TILE) {
    for (int j = threadIdx.x; j < OUT_ROWS * REDUCTION_TILE;
         j += blockDim.x) {
      const int t = chunk_start + tile * OUT_ROWS + j / REDUCTION_TILE;
      lhs[j] = __float2bfloat16_rn(t < tokens
          ? conv[t * conv_dim + inner + groups * SD + group * SD + k + j % REDUCTION_TILE]
          : 0.0f);
    }
    if constexpr (DIRECT_STARTS) {
      for (int j = threadIdx.x; j < REDUCTION_TILE * HD;
           j += blockDim.x) {
        const int dim = j / REDUCTION_TILE;
        const int state = j % REDUCTION_TILE;
        rhs[state * HD + dim] = __float2bfloat16_rn(
            starts_fp32[(((chunk * heads + head) * HD + dim) * SD)
                        + k + state]);
      }
    } else {
      for (int j = threadIdx.x; j < REDUCTION_TILE * HD;
           j += blockDim.x) {
        const int state = k + j / HD, dim = j % HD;
        rhs[j] = starts[(((chunk * heads + head) * SD + state) * HD) + dim];
      }
    }
    __syncthreads();
    for (int sub = 0; sub < REDUCTION_TILE; sub += TILE) {
      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16,
                     wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16,
                     wmma::row_major> bf;
      wmma::load_matrix_sync(
          af, lhs + warp_row * TILE * REDUCTION_TILE + sub,
          REDUCTION_TILE);
      wmma::load_matrix_sync(bf, rhs + sub * HD + warp_col * TILE, HD);
      wmma::mma_sync(state_acc, af, bf, state_acc);
    }
    __syncthreads();
  }
  // The two staging tiles are dead after the final MMA. Reuse their shared
  // storage for one FP32 result at a time, retaining each thread's main
  // values in registers while the state result is written.
  __syncthreads();
  wmma::store_matrix_sync(result + warp_row * TILE * HD + warp_col * TILE,
                          main_acc, HD, wmma::mem_row_major);
  __syncthreads();
  constexpr int VALUES_PER_THREAD = OUT_ROWS * HD / (OUT_ROWS / TILE * HD / TILE * 32);
  float main_values[VALUES_PER_THREAD];
  #pragma unroll
  for (int value = 0; value < VALUES_PER_THREAD; ++value) {
    const int j = int(threadIdx.x) + value * int(blockDim.x);
    main_values[value] = result[j];
  }
  __syncthreads();
  wmma::store_matrix_sync(result + warp_row * TILE * HD + warp_col * TILE,
                          state_acc, HD, wmma::mem_row_major);
  __syncthreads();
  #pragma unroll
  for (int value = 0; value < VALUES_PER_THREAD; ++value) {
    const int j = int(threadIdx.x) + value * int(blockDim.x);
    const int t = chunk_start + tile * OUT_ROWS + j / HD;
    const int dim = j % HD;
    if (t < tokens) {
      const float x = conv[t * conv_dim + head * HD + dim];
      const float initial = reset_at[t * heads + head] < 0.0f
          ? expf(log_decay[t * heads + head]) * result[j] : 0.0f;
      y[t * inner + head * HD + dim] = main_values[value] + initial + x * d[head];
    }
  }
}

template<int REDUCTION_TILE, bool DIRECT_STARTS>
void launch_ssd_output(const float* conv, const float* dt_decay,
                       const float* log_decay, const float* reset_at,
                       const bf16* cb, const bf16* transposed,
                       const float* starts, const float* d, float* y,
                       int tokens, int heads, int groups, int conv_dim,
                       int output_rows, int chunks, cudaStream_t stream) {
  if (output_rows == 128)
    ssd_output<REDUCTION_TILE, DIRECT_STARTS, 128>
        <<<dim3(heads, chunks, CT / 128), 1024, 0, stream>>>(
            conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
            d, y, tokens, heads, groups, conv_dim);
  else
    ssd_output<REDUCTION_TILE, DIRECT_STARTS, 64>
        <<<dim3(heads, chunks, CT / 64), 512, 0, stream>>>(
            conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
            d, y, tokens, heads, groups, conv_dim);
}

extern "C" int xinfer_nemotron_ssd_wmma(
    const float* conv, const float* dt_decay, const float* starts,
    const float* d, float* y, void* cb_raw, float* log_decay,
    float* reset_at, void* transposed_raw,
    int tokens, int heads, int groups, int conv_dim, int cb64, int k64,
    int direct_starts, int output_rows, int output_k128,
    long long stream_raw) {
  if (heads != 64 || groups != 8 || tokens <= 0 ||
      (cb64 != 0 && cb64 != 1) || (k64 != 0 && k64 != 1) ||
      (direct_starts != 0 && direct_starts != 1) ||
      (output_rows != 64 && output_rows != 128) ||
      (output_k128 != 0 && output_k128 != 1) ||
      (output_k128 && !k64) ||
      (!direct_starts && transposed_raw == nullptr))
    return cudaErrorInvalidValue;
  const int chunks = (tokens + CT - 1) / CT;
  auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
  auto cb = static_cast<bf16*>(cb_raw);
  auto transposed = static_cast<bf16*>(transposed_raw);
  if (cb64) {
    if (k64)
      cb_gemm64<64><<<dim3(groups, chunks, 3), 512, 0, stream>>>(
          conv, cb, tokens, heads, groups, conv_dim);
    else
      cb_gemm64<16><<<dim3(groups, chunks, 3), 512, 0, stream>>>(
          conv, cb, tokens, heads, groups, conv_dim);
  } else {
    cb_gemm<<<dim3(groups, chunks, 64), 32, 0, stream>>>(
        conv, cb, tokens, heads, groups, conv_dim);
  }
  decay_prefix<<<dim3(heads, chunks), 32, 0, stream>>>(
      dt_decay, log_decay, reset_at, tokens, heads);
  if (!direct_starts)
    transpose_starts<<<(chunks * heads * SD * HD + 255) / 256, 256, 0, stream>>>(
        starts, transposed, chunks, heads);
  if (k64 && output_k128 && direct_starts)
    launch_ssd_output<128, true>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  else if (k64 && output_k128)
    launch_ssd_output<128, false>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  else if (k64 && direct_starts)
    launch_ssd_output<64, true>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  else if (k64)
    launch_ssd_output<64, false>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  else if (direct_starts)
    launch_ssd_output<16, true>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  else
    launch_ssd_output<16, false>(
        conv, dt_decay, log_decay, reset_at, cb, transposed, starts,
        d, y, tokens, heads, groups, conv_dim, output_rows, chunks, stream);
  return static_cast<int>(cudaPeekAtLastError());
}
