typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

__device__ __forceinline__ float load_bf16(const uint16_t *data, int index) {
  return __uint_as_float(uint32_t(data[index]) << 16);
}

__device__ __forceinline__ void store_bf16(uint16_t *data, int index, float value) {
  uint16_t bits;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  data[index] = bits;
}

// One block owns a token row. Store the residual in BF16 before calculating
// the FP32 norm: the ordinary Candle path rounds the residual at this boundary.
extern "C" __global__ void nemotron_residual_rmsnorm(
    const uint16_t *base, const uint16_t *update,
    const float *weight, uint16_t *residual,
    uint16_t *normalized, int hidden, float eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const int offset = row * hidden;
  float sum = 0.0f;
  for (int col = lane; col < hidden; col += blockDim.x) {
    const float a = load_bf16(base, offset + col);
    const float b = load_bf16(update, offset + col);
    store_bf16(residual, offset + col, a + b);
    const float value = load_bf16(residual, offset + col);
    sum = fmaf(value, value, sum);
  }
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
  }
  if (blockDim.x > 32) {
    __shared__ float warp_sums[32];
    const int warp_id = lane / 32;
    const int warp_lane = lane % 32;
    const int nwarps = blockDim.x / 32;
    if (lane < 32) warp_sums[lane] = 0.0f;
    __syncthreads();
    if (warp_lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    sum = warp_lane < nwarps ? warp_sums[warp_lane] : 0.0f;
    for (int mask = 16; mask > 0; mask >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    if (warp_id == 0 && warp_lane == 0) warp_sums[0] = sum;
    __syncthreads();
    sum = warp_sums[0];
  }
  const float scale = rsqrtf(sum / float(hidden) + eps);
  for (int col = lane; col < hidden; col += blockDim.x) {
    const float value = load_bf16(residual, offset + col);
    store_bf16(normalized, offset + col, (scale * value) * weight[col]);
  }
}
