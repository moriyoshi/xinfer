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

// Eight warps each own one output row. Consecutive lanes read consecutive
// BF16 pairs, so both the weight and input loads are coalesced. Every output
// row is written once; no initialization or atomics are required.
extern "C" __global__ void xinfer_gemma4_bf16_gemv(
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int rows,
    int columns) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;

  const auto* w = reinterpret_cast<const uint32_t*>(
      weight + static_cast<unsigned long long>(row) * columns);
  const auto* x = reinterpret_cast<const uint32_t*>(input);
  float sum = 0.0f;
  for (int column = lane; column < columns / 2; column += 32) {
    const uint32_t a = w[column];
    const uint32_t b = x[column];
    sum = fmaf(bf16_to_f32(a & 0xffffu), bf16_to_f32(b & 0xffffu), sum);
    sum = fmaf(bf16_to_f32(a >> 16), bf16_to_f32(b >> 16), sum);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) output[row] = f32_to_bf16(sum);
}
