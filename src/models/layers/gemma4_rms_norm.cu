typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

__device__ __forceinline__ float bf16_to_f32(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float value) {
  uint16_t bits;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  return bits;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask, 32);
  }
  return value;
}

// Same F32 reduction order as Candle's rmsnorm_f32, but reads BF16 input and
// writes BF16 output directly. One block owns one token row.
extern "C" __global__ void xinfer_gemma4_rms_norm_bf16_f32(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    uint16_t* __restrict__ output,
    int columns,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lanes = blockDim.x;
  float sum = 0.0f;
  for (int col = tid; col < columns; col += lanes) {
    const float value = bf16_to_f32(input[static_cast<unsigned long long>(row) * columns + col]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  if (lanes > 32) {
    __shared__ float warp_sums[32];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < lanes / 32 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (warp == 0 && lane == 0) warp_sums[0] = sum;
    __syncthreads();
    sum = warp_sums[0];
  }
  const float scale = rsqrtf(sum / static_cast<float>(columns) + eps);
  for (int col = tid; col < columns; col += lanes) {
    const unsigned long long index = static_cast<unsigned long long>(row) * columns + col;
    const float value = bf16_to_f32(input[index]);
    output[index] = f32_to_bf16(scale * value * weight[col]);
  }
}

// Preserve BF16 rounding at the norm output before the residual addition.
extern "C" __global__ void xinfer_gemma4_rms_norm_add_bf16_f32(
    const uint16_t* __restrict__ input,
    const float* __restrict__ weight,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ output,
    int columns,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lanes = blockDim.x;
  float sum = 0.0f;
  for (int col = tid; col < columns; col += lanes) {
    const float value = bf16_to_f32(input[static_cast<unsigned long long>(row) * columns + col]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  if (lanes > 32) {
    __shared__ float warp_sums[32];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < lanes / 32 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (warp == 0 && lane == 0) warp_sums[0] = sum;
    __syncthreads();
    sum = warp_sums[0];
  }
  const float scale = rsqrtf(sum / static_cast<float>(columns) + eps);
  for (int col = tid; col < columns; col += lanes) {
    const unsigned long long index = static_cast<unsigned long long>(row) * columns + col;
    const float value = bf16_to_f32(input[index]);
    const float normalized = bf16_to_f32(f32_to_bf16(scale * value * weight[col]));
    output[index] = f32_to_bf16(normalized + bf16_to_f32(residual[index]));
  }
}

// Gemma 4 computes post-attention RMSNorm, adds the residual in BF16, then
// computes pre-feedforward RMSNorm of that rounded sum. Both outputs are used.
extern "C" __global__ void xinfer_gemma4_post_attn_pre_ff_bf16(
    const uint16_t* __restrict__ attention,
    const uint16_t* __restrict__ residual,
    const float* __restrict__ post_weight,
    const float* __restrict__ pre_weight,
    uint16_t* __restrict__ residual_output,
    uint16_t* __restrict__ mlp_input,
    int columns,
    float post_eps,
    float pre_eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lanes = blockDim.x;
  float sum = 0.0f;
  for (int col = tid; col < columns; col += lanes) {
    const float value = bf16_to_f32(attention[static_cast<unsigned long long>(row) * columns + col]);
    sum = fmaf(value, value, sum);
  }
  sum = warp_sum(sum);
  __shared__ float warp_sums[32];
  const int warp = tid / 32;
  const int lane = tid % 32;
  if (lanes > 32) {
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    sum = lane < lanes / 32 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (warp == 0 && lane == 0) warp_sums[0] = sum;
    __syncthreads();
    sum = warp_sums[0];
  }
  const float post_scale = rsqrtf(sum / static_cast<float>(columns) + post_eps);
  float pre_sum = 0.0f;
  for (int col = tid; col < columns; col += lanes) {
    const unsigned long long index = static_cast<unsigned long long>(row) * columns + col;
    const float value = bf16_to_f32(attention[index]);
    const float normalized = bf16_to_f32(f32_to_bf16(post_scale * value * post_weight[col]));
    const uint16_t added = f32_to_bf16(normalized + bf16_to_f32(residual[index]));
    residual_output[index] = added;
    const float rounded = bf16_to_f32(added);
    pre_sum = fmaf(rounded, rounded, pre_sum);
  }
  pre_sum = warp_sum(pre_sum);
  if (lanes > 32) {
    if (tid < 32) warp_sums[tid] = 0.0f;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = pre_sum;
    __syncthreads();
    pre_sum = lane < lanes / 32 ? warp_sums[lane] : 0.0f;
    pre_sum = warp_sum(pre_sum);
    if (warp == 0 && lane == 0) warp_sums[0] = pre_sum;
    __syncthreads();
    pre_sum = warp_sums[0];
  }
  const float pre_scale = rsqrtf(pre_sum / static_cast<float>(columns) + pre_eps);
  for (int col = tid; col < columns; col += lanes) {
    const unsigned long long index = static_cast<unsigned long long>(row) * columns + col;
    const float rounded = bf16_to_f32(residual_output[index]);
    mlp_input[index] = f32_to_bf16(pre_scale * rounded * pre_weight[col]);
  }
}
