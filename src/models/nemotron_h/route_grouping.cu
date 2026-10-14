typedef unsigned int uint32_t;

struct Params {
    int slots;
    int experts;
};

// One block scans the assignment IDs for each expert. No cross-block atomics
// are needed; a later prefix pass determines each expert's output range.
extern "C" __global__ void nemotron_route_count(
    const uint32_t* ids, uint32_t* counts, Params p) {
    const int expert = int(blockIdx.x);
    const int lane = int(threadIdx.x);
    __shared__ uint32_t partial[256];
    uint32_t count = 0;
    for (int i = lane; i < p.slots; i += 256) {
        count += ids[i] == uint32_t(expert);
    }
    partial[lane] = count;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) counts[expert] = partial[0];
}

extern "C" __global__ void nemotron_route_prefix(
    const uint32_t* counts, uint32_t* offsets, Params p) {
    uint32_t running = 0;
    offsets[0] = 0;
    for (int expert = 0; expert < p.experts; ++expert) {
        running += counts[expert];
        offsets[expert + 1] = running;
    }
}

// Every block owns exactly one expert's output range. Warp ballots compute
// each matching assignment's stable rank within a 256-element input tile.
extern "C" __global__ void nemotron_route_scatter(
    const uint32_t* ids, const uint32_t* offsets,
    uint32_t* sorted_experts, uint32_t* sorted_assignments, Params p) {
    const int expert = int(blockIdx.x);
    const int thread = int(threadIdx.x);
    const int warp = thread >> 5;
    const int lane = thread & 31;
    __shared__ uint32_t warp_counts[8];
    uint32_t running = offsets[expert];
    for (int base = 0; base < p.slots; base += 256) {
        const int assignment = base + thread;
        const bool selected = assignment < p.slots && ids[assignment] == uint32_t(expert);
        const uint32_t mask = __ballot_sync(0xffffffffu, selected);
        if (lane == 0) warp_counts[warp] = __popc(mask);
        __syncthreads();
        uint32_t earlier_warps = 0;
        #pragma unroll
        for (int w = 0; w < warp; ++w) earlier_warps += warp_counts[w];
        const uint32_t earlier_lanes = __popc(mask & ((1u << lane) - 1u));
        if (selected) {
            const uint32_t output = running + earlier_warps + earlier_lanes;
            sorted_experts[output] = uint32_t(expert);
            sorted_assignments[output] = uint32_t(assignment);
        }
        uint32_t tile_count = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tile_count += warp_counts[w];
        running += tile_count;
        __syncthreads();
    }
}

struct CombineParams {
    int rows;
    int topk;
    int hidden;
    int dtype; // 0 BF16, 1 F16
};

__device__ __forceinline__ float moe_half_to_float(unsigned short bits) {
    const uint32_t sign = uint32_t(bits & 0x8000u) << 16;
    int exponent = (bits >> 10) & 31;
    uint32_t mantissa = bits & 0x3ffu;
    if (exponent == 0 && mantissa != 0) {
        exponent = 1;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x3ffu;
    }
    const uint32_t raw = exponent == 0
        ? sign
        : exponent == 31
            ? sign | 0x7f800000u | (mantissa << 13)
            : sign | (uint32_t(exponent + 112) << 23) | (mantissa << 13);
    return __uint_as_float(raw);
}

__device__ __forceinline__ float moe_load(const unsigned short* data, int index, int dtype) {
    const unsigned short bits = data[index];
    return dtype == 0 ? __uint_as_float(uint32_t(bits) << 16) : moe_half_to_float(bits);
}

__device__ __forceinline__ void moe_store(unsigned short* data, int index, float value, int dtype) {
    unsigned short bits;
    if (dtype == 0) {
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    } else {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    }
    data[index] = bits;
}

// Preserve the original FP32 route multiplication, top-k sum order, shared
// addition, and final BF16/F16 rounding without materializing six wide
// intermediate tensors for each MoE layer.
extern "C" __global__ void nemotron_route_combine(
    const unsigned short* routed, const unsigned short* shared,
    const float* weights, unsigned short* output, CombineParams p) {
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= p.rows * p.hidden) return;
    const int row = index / p.hidden;
    const int column = index % p.hidden;
    float sum = 0.0f;
    for (int k = 0; k < p.topk; ++k) {
        const float value = moe_load(routed,
            (row * p.topk + k) * p.hidden + column, p.dtype);
        const float weighted = __fmul_rn(value, weights[row * p.topk + k]);
        sum = __fadd_rn(sum, weighted);
    }
    const float combined = __fadd_rn(sum, moe_load(shared, index, p.dtype));
    moe_store(output, index, combined, p.dtype);
}
