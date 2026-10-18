typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

__device__ __forceinline__ float as_f32(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t as_bf16(float x) {
  uint16_t bits;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
  return bits;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, mask, 32);
  return value;
}

// One block owns one token/head. The reduction and RoPE remain in F32; the
// only rounding is the final BF16 store, as in the original Gemma 4 path.
extern "C" __global__ void xinfer_gemma4_qk_norm_rope(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int64_t* __restrict__ positions,
    uint16_t* __restrict__ q_out,
    uint16_t* __restrict__ k_out,
    int q_heads, int k_heads, int width, float q_eps, float k_eps,
    int interleaved) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const bool is_q = head < q_heads;
  const int local_head = is_q ? head : head - q_heads;
  const int heads = is_q ? q_heads : k_heads;
  const uint16_t* input = is_q ? q : k;
  uint16_t* output = is_q ? q_out : k_out;
  const float* weight = is_q ? q_weight : k_weight;
  const float eps = is_q ? q_eps : k_eps;
  const uint64_t row = (static_cast<uint64_t>(token) * heads + local_head) * width;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int col = tid; col < width; col += blockDim.x) {
    const float x = as_f32(input[row + col]);
    sum = fmaf(x, x, sum);
  }
  sum = warp_sum(sum);
  if (blockDim.x > 32) {
    __shared__ float warp_sums[32];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (warp == 0 && lane == 0) warp_sums[0] = sum;
    __syncthreads();
    sum = warp_sums[0];
  }
  const float scale = rsqrtf(sum / static_cast<float>(width) + eps);
  const uint64_t cs_row = static_cast<uint64_t>(positions[token]) * (width / 2);
  for (int pair = tid; pair < width / 2; pair += blockDim.x) {
    const int x_col = interleaved ? 2 * pair : pair;
    const int y_col = interleaved ? 2 * pair + 1 : pair + width / 2;
    const float x = scale * as_f32(input[row + x_col]) * weight[x_col];
    const float y = scale * as_f32(input[row + y_col]) * weight[y_col];
    const float c = cos[cs_row + pair];
    const float s = sin[cs_row + pair];
    output[row + x_col] = as_bf16(x * c - y * s);
    output[row + y_col] = as_bf16(y * c + x * s);
  }
}
