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

// Input is the contiguous [tokens, gate || up] output of the merged NVFP4
// projection. The BF16 rounding after GELU matches Candle's separate GELU
// followed by BF16 multiply.
extern "C" __global__ void xinfer_gemma4_gated_gelu_bf16(
    const uint16_t* __restrict__ gate_up,
    uint16_t* __restrict__ output,
    unsigned long long elements,
    int width) {
  const unsigned long long start =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long i = start; i < elements; i += stride) {
    const unsigned long long row = i / width;
    const unsigned long long column = i - row * width;
    const unsigned long long base = row * (2ULL * width);
    const float gate = bf16_to_f32(gate_up[base + column]);
    const float up = bf16_to_f32(gate_up[base + width + column]);
    const float gelu = 0.5f * gate *
        (1.0f + tanhf(0.7978845608f *
                       (gate + 0.044715f * gate * gate * gate)));
    const float rounded_gelu = bf16_to_f32(f32_to_bf16(gelu));
    output[i] = f32_to_bf16(rounded_gelu * up);
  }
}
