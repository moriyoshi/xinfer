// Fused routed-row gather and NVFP4 activation quantization for Nemotron-H.
// The conversion and scale swizzle match attention-rs c0f19f2 (MIT); only
// source-row addressing changes. Requires sm_121a FP4 instructions.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

static constexpr int NVFP4_BLOCK_SIZE = 16;
// Custom SM100 quant path: clamp E4M3 block scale to 126. SM120/SM121 uses
// FlashInfer invokeFP4Quantization (e4m3Max=448) instead of this kernel.
static constexpr uint8_t NVFP4_E4M3_SCALE_MAX = 126;

__device__ __forceinline__ uint8_t clamp_nvfp4_e4m3_scale(uint8_t bits) {
  return bits > NVFP4_E4M3_SCALE_MAX ? NVFP4_E4M3_SCALE_MAX : bits;
}

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
  float abs_val = fabsf(val);
  uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;

  // Round-to-nearest-even at midpoints between representable FP4 E2M1 values.
  // Representable: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (codes 0-7).
  // At exact midpoints, round to the value with an even code (0,2,4,6).
  uint8_t code;
  if (abs_val <= 0.25f) {
    code = 0x0;  // 0.0  (midpoint 0.25 rounds to even code 0)
  } else if (abs_val < 0.75f) {
    code = 0x1;  // 0.5
  } else if (abs_val <= 1.25f) {
    code = 0x2;  // 1.0  (midpoint 0.75→1.0 already correct; 1.25 rounds to even code 2)
  } else if (abs_val < 1.75f) {
    code = 0x3;  // 1.5
  } else if (abs_val <= 2.5f) {
    code = 0x4;  // 2.0  (midpoint 1.75→2.0 already correct; 2.5 rounds to even code 4)
  } else if (abs_val < 3.5f) {
    code = 0x5;  // 3.0
  } else if (abs_val <= 5.0f) {
    code = 0x6;  // 4.0  (midpoint 3.5→4.0 already correct; 5.0 rounds to even code 6)
  } else {
    code = 0x7;  // 6.0
  }

  return sign | code;
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(float (&vals)[8]) {
  uint32_t result = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    result |= static_cast<uint32_t>(float_to_fp4_e2m1(vals[i])) << (4 * i);
  }
  return result;
}

__device__ __forceinline__ int nvfp4_find_expert_for_row(
    const int32_t* __restrict__ expert_offsets,
    int num_experts,
    int row) {
  int lo = 0;
  int hi = num_experts;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (expert_offsets[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename InType, bool ReluSquared>
__global__ void xinfer_nvfp4_quantize_gathered_kernel(
    const InType* __restrict__ input,        // [total_rows, K]
    uint8_t* __restrict__ output,            // [total_rows, K/2]
    uint8_t* __restrict__ swizzled_scales,   // flat swizzled buffer
    const int32_t* __restrict__ sorted_token_ids,
    int map_divisor,
    const float* __restrict__ input_scale_invs,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sf_offsets,
    int total_rows,
    int num_experts,
    int K,
    int K_scale_padded) {
  int row = blockIdx.x;
  int num_blocks = K / NVFP4_BLOCK_SIZE;
  if (row >= total_rows) {
    return;
  }

  int expert_id = nvfp4_find_expert_for_row(expert_offsets, num_experts, row);
  int local_row = row - expert_offsets[expert_id];
  int local_row_padded_base = sf_offsets[expert_id];
  int rows = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
  int rows_padded = ((rows + 127) / 128) * 128;
  float SFScaleVal = input_scale_invs[expert_id];

  for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {

  int k_start = block_idx * NVFP4_BLOCK_SIZE;
  int64_t source_row = static_cast<int64_t>(sorted_token_ids[row]) / map_divisor;
  int64_t in_base = source_row * K + k_start;
  float vals[16];
  for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
    float value = static_cast<float>(input[in_base + i]);
    if constexpr (ReluSquared) {
      // The unfused Nemotron path rounds the ReLU square to BF16 before
      // activation quantization. Preserve that rounding at the source read.
      float positive = value < 0.0f ? 0.0f : value;
      value = static_cast<float>(InType(__fmul_rn(positive, positive)));
    }
    vals[i] = value;
  }

  float vecMax = 0.0f;
  for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
    vecMax = fmaxf(vecMax, fabsf(vals[i]));
  }

  int64_t out_base = static_cast<int64_t>(row) * (K / 2) + k_start / 2;


  float SFValue = __fdiv_rn(SFScaleVal * vecMax, 6.0f);

  __nv_fp8_e4m3 fp8_sf = __nv_fp8_e4m3(SFValue);
  uint8_t fp8_scale_bits = clamp_nvfp4_e4m3_scale(fp8_sf.__x);
  if (fp8_scale_bits != fp8_sf.__x) {
    fp8_sf.__x = fp8_scale_bits;
    SFValue = static_cast<float>(fp8_sf);
  } else {
    SFValue = static_cast<float>(fp8_sf);
  }

  float outputScale = SFValue != 0.0f
      ? __fdiv_rn(SFScaleVal, SFValue)
      : 0.0f;

  float scaled_vals_0[8], scaled_vals_1[8];
  for (int i = 0; i < 8; ++i) {
    scaled_vals_0[i] = vals[i] * outputScale;
    scaled_vals_1[i] = vals[8 + i] * outputScale;
  }

  uint32_t packed_lo = fp32x8_to_e2m1x8(scaled_vals_0);
  uint32_t packed_hi = fp32x8_to_e2m1x8(scaled_vals_1);
  reinterpret_cast<uint32_t*>(output + out_base)[0] = packed_lo;
  reinterpret_cast<uint32_t*>(output + out_base)[1] = packed_hi;


  int innerKIdx = block_idx % 4;
  int innerMIdx = (local_row % 128) / 32;
  int outerMIdx = local_row % 32;
  int kTileIdx = block_idx / 4;
  int mTileIdx = local_row / 128;
  int numKTiles = K_scale_padded / 4;
  int64_t kTileStride = 512;
  int64_t mTileStride = static_cast<int64_t>(numKTiles) * kTileStride;
  int64_t chunk_base = static_cast<int64_t>(local_row_padded_base) * K_scale_padded;
  int64_t dstOffset = chunk_base +
      static_cast<int64_t>(mTileIdx) * mTileStride +
      static_cast<int64_t>(kTileIdx) * kTileStride +
      outerMIdx * 16 +
      innerMIdx * 4 +
      innerKIdx;

  if (local_row < rows && local_row < rows_padded) {
    swizzled_scales[dstOffset] = fp8_scale_bits;
  }

  } // for block_idx
}


extern "C" void xinfer_nvfp4_quantize_gathered_bf16(
    const void* input, void* output, void* scales,
    const int32_t* sorted_ids, int map_divisor,
    const float* input_scale_invs, const int32_t* expert_offsets,
    const int32_t* sf_offsets, int rows, int experts,
    int K, int K_scale_padded, int relu_squared, int64_t stream) {
  dim3 grid(rows);
  dim3 block(min(K / 16, 512));
  if (relu_squared) {
    xinfer_nvfp4_quantize_gathered_kernel<nv_bfloat16, true>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const nv_bfloat16*>(input), static_cast<uint8_t*>(output),
            static_cast<uint8_t*>(scales), sorted_ids, map_divisor,
            input_scale_invs, expert_offsets, sf_offsets,
            rows, experts, K, K_scale_padded);
  } else {
    xinfer_nvfp4_quantize_gathered_kernel<nv_bfloat16, false>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const nv_bfloat16*>(input), static_cast<uint8_t*>(output),
            static_cast<uint8_t*>(scales), sorted_ids, map_divisor,
            input_scale_invs, expert_offsets, sf_offsets,
            rows, experts, K, K_scale_padded);
  }
}

extern "C" void xinfer_nvfp4_quantize_gathered_f16(
    const void* input, void* output, void* scales,
    const int32_t* sorted_ids, int map_divisor,
    const float* input_scale_invs, const int32_t* expert_offsets,
    const int32_t* sf_offsets, int rows, int experts,
    int K, int K_scale_padded, int relu_squared, int64_t stream) {
  dim3 grid(rows);
  dim3 block(min(K / 16, 512));
  if (relu_squared) {
    xinfer_nvfp4_quantize_gathered_kernel<half, true>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const half*>(input), static_cast<uint8_t*>(output),
            static_cast<uint8_t*>(scales), sorted_ids, map_divisor,
            input_scale_invs, expert_offsets, sf_offsets,
            rows, experts, K, K_scale_padded);
  } else {
    xinfer_nvfp4_quantize_gathered_kernel<half, false>
        <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const half*>(input), static_cast<uint8_t*>(output),
            static_cast<uint8_t*>(scales), sorted_ids, map_divisor,
            input_scale_invs, expert_offsets, sf_offsets,
            rows, experts, K, K_scale_padded);
  }
}

// The grouped GEMM returns FP32 rows in expert order. Each route owns one
// distinct destination row, so a whole row can be converted and weighted by
// one block with coalesced float4 reads and 64-bit BF16 writes. This retains
// the reference's FP32 route multiplication followed by one BF16 rounding.
__global__ void xinfer_nvfp4_scatter_f32_to_bf16_vec_kernel(
    const float* __restrict__ input, const int32_t* __restrict__ sorted_ids,
    nv_bfloat16* __restrict__ output, const float* __restrict__ weights,
    int rows, int columns) {
  const int source_row = blockIdx.x;
  if (source_row >= rows) return;
  const int destination_row = sorted_ids[source_row];
  if (destination_row < 0 || destination_row >= rows) return;
  const float weight = weights == nullptr ? 1.0f : weights[destination_row];
  const float4* source = reinterpret_cast<const float4*>(
      input + static_cast<int64_t>(source_row) * columns);
  ushort4* destination = reinterpret_cast<ushort4*>(
      output + static_cast<int64_t>(destination_row) * columns);
  for (int vector = threadIdx.x; vector < columns / 4; vector += blockDim.x) {
    const float4 value = source[vector];
    ushort4 packed;
    packed.x = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value.x, weight)));
    packed.y = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value.y, weight)));
    packed.z = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value.z, weight)));
    packed.w = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value.w, weight)));
    destination[vector] = packed;
  }
}

extern "C" int xinfer_nvfp4_scatter_f32_to_bf16_vec(
    const float* input, const int32_t* sorted_ids, void* output,
    const float* weights, int rows, int columns, int64_t stream) {
  if (rows <= 0 || columns <= 0 || columns % 4 != 0) return -1;
  xinfer_nvfp4_scatter_f32_to_bf16_vec_kernel<<<rows, 256, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
          input, sorted_ids, static_cast<nv_bfloat16*>(output), weights,
          rows, columns);
  return static_cast<int>(cudaGetLastError());
}

// sorted_ids maps grouped-GEMM rows to logical route assignments. The native
// route grouping guarantees a permutation, so every inverse slot is written
// once and the finalizer may read it on the same stream.
__global__ void xinfer_nvfp4_invert_routes_kernel(
    const int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ inverse, int assignments) {
  const int source = blockIdx.x * blockDim.x + threadIdx.x;
  if (source >= assignments) return;
  const int destination = sorted_ids[source];
  if (destination >= 0 && destination < assignments) inverse[destination] = source;
}

// Match the existing two-kernel path: round each unweighted expert result to
// BF16, multiply by its FP32 route weight, sum in top-k order, add the shared
// expert in FP32, then round the combined output to BF16 once.
__global__ void xinfer_nvfp4_finalize_bf16_kernel(
    const float* __restrict__ routed,
    const int32_t* __restrict__ inverse,
    const nv_bfloat16* __restrict__ shared,
    const float* __restrict__ weights,
    nv_bfloat16* __restrict__ output,
    int rows, int topk, int hidden) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * hidden) return;
  const int row = index / hidden;
  const int column = index % hidden;
  float sum = 0.0f;
  for (int k = 0; k < topk; ++k) {
    const int assignment = row * topk + k;
    const float rounded = __bfloat162float(__float2bfloat16_rn(
        routed[static_cast<int64_t>(inverse[assignment]) * hidden + column]));
    sum = __fadd_rn(sum, __fmul_rn(rounded, weights[assignment]));
  }
  const float combined = __fadd_rn(sum, __bfloat162float(shared[index]));
  output[index] = __float2bfloat16_rn(combined);
}

extern "C" int xinfer_nvfp4_finalize_bf16(
    const float* routed, const int32_t* sorted_ids, int32_t* inverse,
    const void* shared, const float* weights, void* output,
    int rows, int topk, int hidden, int64_t stream) {
  if (rows <= 0 || topk <= 0 || topk > 8 || hidden <= 0 ||
      static_cast<int64_t>(rows) * topk > INT32_MAX ||
      static_cast<int64_t>(rows) * hidden > INT32_MAX) return -1;
  const int assignments = rows * topk;
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  xinfer_nvfp4_invert_routes_kernel<<<(assignments + 255) / 256, 256, 0, cuda_stream>>>(
      sorted_ids, inverse, assignments);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  xinfer_nvfp4_finalize_bf16_kernel<<<(rows * hidden + 255) / 256, 256, 0,
      cuda_stream>>>(routed, inverse, static_cast<const nv_bfloat16*>(shared),
          weights, static_cast<nv_bfloat16*>(output), rows, topk, hidden);
  return static_cast<int>(cudaGetLastError());
}
