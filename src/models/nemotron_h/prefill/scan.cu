typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

#ifndef SUMMARY_WARPS
#define SUMMARY_WARPS 4
#endif
#define SCAN_WARPS 8

struct Params {
    int tokens;
    int heads;
    int head_dim;
    int groups;
    int state_dim;
    int kernel;
    int conv_dim;
    int in_dim;
    int projected_dtype;
    int output_dtype;
    float dt_min;
    float dt_max;
    float eps;
};

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
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
    const uint32_t f32_bits = exponent == 0
        ? sign
        : exponent == 31
            ? sign | 0x7f800000u | (mantissa << 13)
            : sign | (uint32_t(exponent + 112) << 23) | (mantissa << 13);
    return __uint_as_float(f32_bits);
}

__device__ __forceinline__ void store_projected(void* output, int index, float value, int dtype) {
    if (dtype == 0) {
        uint16_t bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
        static_cast<uint16_t*>(output)[index] = bits;
    } else if (dtype == 1) {
        uint16_t bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
        static_cast<uint16_t*>(output)[index] = bits;
    } else {
        static_cast<float*>(output)[index] = value;
    }
}

__device__ __forceinline__ float projected_at(const void* projected, int index, int dtype) {
    if (dtype == 0) {
        return __uint_as_float(uint32_t(static_cast<const uint16_t*>(projected)[index]) << 16);
    }
    if (dtype == 1) {
        return half_bits_to_float(static_cast<const uint16_t*>(projected)[index]);
    }
    return static_cast<const float*>(projected)[index];
}

// The old per-token window is [history, current input]. This kernel reads the
// same causal window for each token without changing the carried history until
// every row is prepared. It also prepares dt and the final history in one pass.
extern "C" __global__ void nemotron_mamba_prepare(
    const void* projected, const float* history, const float* weight,
    const float* bias, const float* dt_bias, const float* a,
    float* conv, float* next_history, float* dt_decay, Params p) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int conv_work = p.tokens * p.conv_dim;
    const int history_work = (p.kernel - 1) * p.conv_dim;
    const int inner = p.heads * p.head_dim;
    if (i < conv_work) {
        const int t = i / p.conv_dim;
        const int channel = i % p.conv_dim;
        float sum = 0.0f;
        for (int k = 0; k < p.kernel; ++k) {
            const int position = t + k - (p.kernel - 1);
            const float input = position < 0
                ? history[(position + p.kernel - 1) * p.conv_dim + channel]
                : projected_at(projected, position * p.in_dim + inner + channel, p.projected_dtype);
            sum += input * weight[k * p.conv_dim + channel];
        }
        const float value = sum + bias[channel];
        conv[i] = value / (1.0f + expf(-value));
    } else if (i < conv_work + history_work) {
        const int index = i - conv_work;
        const int row = index / p.conv_dim;
        const int channel = index % p.conv_dim;
        const int position = p.tokens + row - (p.kernel - 1);
        next_history[index] = position < 0
            ? history[(position + p.kernel - 1) * p.conv_dim + channel]
            : projected_at(projected, position * p.in_dim + inner + channel, p.projected_dtype);
    } else if (i < conv_work + history_work + p.tokens * p.heads) {
        const int index = i - conv_work - history_work;
        const int t = index / p.heads;
        const int head = index % p.heads;
        const float raw = projected_at(
            projected, t * p.in_dim + inner + p.conv_dim + head, p.projected_dtype
        ) + dt_bias[head];
        const float dt = fminf(fmaxf(fmaxf(raw, 0.0f) + log1pf(expf(-fabsf(raw))), p.dt_min), p.dt_max);
        dt_decay[index * 2] = dt;
        dt_decay[index * 2 + 1] = expf(dt * a[head]);
    }
}

// A block owns one head. Each warp owns one head-dimension row at a time;
// each lane holds four state channels. The full state stays in registers over
// the token scan. Warp reduction fixes summation order and uses no atomics.
extern "C" __global__ void nemotron_mamba_scan(
    const float* conv, const float* dt_decay, const float* initial_state,
    const float* d, float* y, float* final_state, Params p) {
    const int head = int(blockIdx.x);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int channel = lane * 4;
    const int rows = HEAD_DIM / 8;
    float state[HEAD_DIM / 8][4];

    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        const int row = r * 8 + warp;
        const int offset = (head * HEAD_DIM + row) * 128 + channel;
        #pragma unroll
        for (int j = 0; j < 4; ++j) state[r][j] = initial_state[offset + j];
    }

    const int group = head / (p.heads / p.groups);
    const int inner = p.heads * HEAD_DIM;
    const int group_base = inner + group * 128 + channel;
    for (int t = 0; t < p.tokens; ++t) {
        const float dt = dt_decay[(t * p.heads + head) * 2];
        const float decay = dt_decay[(t * p.heads + head) * 2 + 1];
        const int conv_base = t * p.conv_dim;
        float b[4], c[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            b[j] = conv[conv_base + group_base + j];
            c[j] = conv[conv_base + group_base + p.groups * 128 + j];
        }
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = r * 8 + warp;
            const float x = conv[conv_base + head * HEAD_DIM + row];
            const float scaled_x = dt * x;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[r][j] = decay * state[r][j] + scaled_x * b[j];
                sum += state[r][j] * c[j];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) {
                y[t * inner + head * HEAD_DIM + row] = sum + x * d[head];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        const int row = r * 8 + warp;
        const int offset = (head * HEAD_DIM + row) * 128 + channel;
        #pragma unroll
        for (int j = 0; j < 4; ++j) final_state[offset + j] = state[r][j];
    }
}

// Each block summarizes one short, independent head/chunk interval as an
// affine recurrence: final_state = alpha * initial_state + summary. The
// summary starts from zero; a second pass supplies the true chunk inputs.
extern "C" __global__ void nemotron_mamba_chunk_summary(
    const float* conv, const float* dt_decay, float* alphas,
    float* summaries, int chunk_size, Params p) {
    const int head = int(blockIdx.x);
    const int chunk = int(blockIdx.y);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int channel = lane * 4;
    const int rows = HEAD_DIM / SUMMARY_WARPS;
    const int start = chunk * chunk_size;
    const int end = min(start + chunk_size, p.tokens);
    const int inner = p.heads * HEAD_DIM;
    const int group = head / (p.heads / p.groups);
    const int group_base = inner + group * 128 + channel;
    float state[HEAD_DIM / SUMMARY_WARPS][4];
    float alpha = 1.0f;
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) state[r][j] = 0.0f;
    }
    for (int t = start; t < end; ++t) {
        const float dt = dt_decay[(t * p.heads + head) * 2];
        const float decay = dt_decay[(t * p.heads + head) * 2 + 1];
        alpha *= decay;
        const int conv_base = t * p.conv_dim;
        float b[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = conv[conv_base + group_base + j];
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = r * SUMMARY_WARPS + warp;
            const float x = conv[conv_base + head * HEAD_DIM + row];
            const float scaled_x = dt * x;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[r][j] = decay * state[r][j] + scaled_x * b[j];
            }
        }
    }
    const int block_state = (chunk * p.heads + head) * HEAD_DIM * 128;
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        const int row = r * SUMMARY_WARPS + warp;
        const int offset = block_state + row * 128 + channel;
        #pragma unroll
        for (int j = 0; j < 4; ++j) summaries[offset + j] = state[r][j];
    }
    if (threadIdx.x == 0) alphas[chunk * p.heads + head] = alpha;
}

// Only chunk boundaries are sequential. The work within every chunk runs in
// parallel in the next kernel, and no host synchronization is needed.
extern "C" __global__ void nemotron_mamba_chunk_prefix(
    const float* initial_state, const float* alphas,
    float* summaries_and_starts, int chunks, Params p) {
    const int head = int(blockIdx.x);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int channel = lane * 4;
    const int rows = HEAD_DIM / SUMMARY_WARPS;
    float state[HEAD_DIM / SUMMARY_WARPS][4];
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        const int row = r * SUMMARY_WARPS + warp;
        const int offset = (head * HEAD_DIM + row) * 128 + channel;
        #pragma unroll
        for (int j = 0; j < 4; ++j) state[r][j] = initial_state[offset + j];
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float alpha = alphas[chunk * p.heads + head];
        const int block_state = (chunk * p.heads + head) * HEAD_DIM * 128;
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = r * SUMMARY_WARPS + warp;
            const int offset = block_state + row * 128 + channel;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float summary = summaries_and_starts[offset + j];
                summaries_and_starts[offset + j] = state[r][j];
                state[r][j] = alpha * state[r][j] + summary;
            }
        }
    }
}

extern "C" __global__ void nemotron_mamba_chunk_scan(
    const float* conv, const float* dt_decay, const float* starts,
    const float* d, float* y, float* final_state, int chunk_size, Params p) {
    const int head = int(blockIdx.x);
    const int chunk = int(blockIdx.y);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int channel = lane * 4;
    const int rows = HEAD_DIM / SCAN_WARPS;
    const int start = chunk * chunk_size;
    const int end = min(start + chunk_size, p.tokens);
    const int chunks = (p.tokens + chunk_size - 1) / chunk_size;
    const int group = head / (p.heads / p.groups);
    const int inner = p.heads * HEAD_DIM;
    const int group_base = inner + group * 128 + channel;
    const int block_state = (chunk * p.heads + head) * HEAD_DIM * 128;
    float state[HEAD_DIM / SCAN_WARPS][4];
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        const int row = r * SCAN_WARPS + warp;
        const int offset = block_state + row * 128 + channel;
        #pragma unroll
        for (int j = 0; j < 4; ++j) state[r][j] = starts[offset + j];
    }
    for (int t = start; t < end; ++t) {
        const float dt = dt_decay[(t * p.heads + head) * 2];
        const float decay = dt_decay[(t * p.heads + head) * 2 + 1];
        const int conv_base = t * p.conv_dim;
        float b[4], c[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            b[j] = conv[conv_base + group_base + j];
            c[j] = conv[conv_base + group_base + p.groups * 128 + j];
        }
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = r * SCAN_WARPS + warp;
            const float x = conv[conv_base + head * HEAD_DIM + row];
            const float scaled_x = dt * x;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[r][j] = decay * state[r][j] + scaled_x * b[j];
                sum += state[r][j] * c[j];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) y[t * inner + head * HEAD_DIM + row] = sum + x * d[head];
        }
    }
    if (chunk == chunks - 1) {
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = r * SCAN_WARPS + warp;
            const int offset = (head * HEAD_DIM + row) * 128 + channel;
            #pragma unroll
            for (int j = 0; j < 4; ++j) final_state[offset + j] = state[r][j];
        }
    }
}

// Prefill epilogue: gate, grouped RMSNorm and output conversion in one pass.
// Each block owns one token/group, so reductions are deterministic and no
// intermediate BF16/F32 tensors or additional device round trips are needed.
extern "C" __global__ void nemotron_mamba_gate_norm(
    const void* projected, const float* y, const float* norm_weight,
    void* normalized, Params p) {
    const int group = int(blockIdx.x);
    const int token = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int inner = p.heads * p.head_dim;
    const int group_rows = inner / p.groups;
    const int row_base = group * group_rows;
    extern __shared__ float scratch[];
    float* gated = scratch;
    float* warp_sums = scratch + group_rows;
    for (int row = lane; row < group_rows; row += blockDim.x) {
        const int index = row_base + row;
        const float gate = projected_at(
            projected, token * p.in_dim + index, p.projected_dtype);
        const float value = y[token * inner + index]
            * (gate / (1.0f + expf(-gate)));
        gated[row] = value;
    }
    __syncthreads();
    // Match Candle fast_sum_f32's vectorized float4 and xor-warp reduction
    // order. The gate values live in shared memory rather than an F32 tensor.
    float local_sum = 0.0f;
    for (int vec = lane; vec < group_rows / 4; vec += blockDim.x) {
        const int i = vec * 4;
        const float a = __fmul_rn(gated[i], gated[i]);
        const float b = __fmul_rn(gated[i + 1], gated[i + 1]);
        const float c = __fmul_rn(gated[i + 2], gated[i + 2]);
        const float d = __fmul_rn(gated[i + 3], gated[i + 3]);
        local_sum += a + b + c + d;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
    const int warp = lane >> 5;
    if ((lane & 31) == 0) warp_sums[warp] = local_sum;
    __syncthreads();
    if (lane < 32) {
        local_sum = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
        if (lane == 0) warp_sums[0] = local_sum;
    }
    __syncthreads();
    const float variance = warp_sums[0] * (1.0f / float(group_rows));
    const float denom = sqrtf(variance + p.eps);
    for (int row = lane; row < group_rows; row += blockDim.x) {
        const int index = row_base + row;
        store_projected(normalized, token * inner + index,
            (gated[row] / denom) * norm_weight[index], p.output_dtype);
    }
}

// Single-token decode. A block owns one Mamba group and writes its disjoint
// conv/SSM state to fresh buffers. The block also reduces its grouped RMSNorm, so the
// only subsequent operation is out_proj's GEMV. No atomics or host reads.
extern "C" __global__ void nemotron_mamba_step(
    const void* projected, const float* history, float* next_history, const float* weight,
    const float* bias, const float* dt_bias, const float* a, const float* d,
    const float* norm_weight, const float* state, float* next_state,
    void* normalized, Params p) {
    const int group = int(blockIdx.x);
    const int lane = int(threadIdx.x);
    const int inner = p.heads * p.head_dim;
    const int heads_per_group = p.heads / p.groups;
    const int group_rows = heads_per_group * p.head_dim;
    extern __shared__ float scratch[];
    float* bc = scratch;                       // 128 B + 128 C values
    float* gated = scratch + 256;             // group_rows values
    float* reduction = gated + group_rows;    // 256 values

    const int bc_channel = lane < 128
        ? inner + group * 128 + lane
        : inner + p.groups * 128 + group * 128 + lane - 128;
    float conv_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < p.kernel - 1; ++k) {
        conv_sum += history[k * p.conv_dim + bc_channel] * weight[k * p.conv_dim + bc_channel];
    }
    const float bc_input = projected_at(projected, inner + bc_channel, p.projected_dtype);
    conv_sum += bc_input * weight[(p.kernel - 1) * p.conv_dim + bc_channel];
    conv_sum += bias[bc_channel];
    bc[lane] = conv_sum / (1.0f + expf(-conv_sum));
    for (int k = 0; k < p.kernel - 2; ++k) {
        next_history[k * p.conv_dim + bc_channel] = history[(k + 1) * p.conv_dim + bc_channel];
    }
    next_history[(p.kernel - 2) * p.conv_dim + bc_channel] = bc_input;
    __syncthreads();

    for (int local_row = lane; local_row < group_rows; local_row += blockDim.x) {
        const int head = group * heads_per_group + local_row / p.head_dim;
        const int global_row = head * p.head_dim + local_row % p.head_dim;
        float x_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < p.kernel - 1; ++k) {
            x_sum += history[k * p.conv_dim + global_row] * weight[k * p.conv_dim + global_row];
        }
        const float x_input = projected_at(projected, inner + global_row, p.projected_dtype);
        x_sum += x_input * weight[(p.kernel - 1) * p.conv_dim + global_row];
        x_sum += bias[global_row];
        const float x = x_sum / (1.0f + expf(-x_sum));
        for (int k = 0; k < p.kernel - 2; ++k) {
            next_history[k * p.conv_dim + global_row] = history[(k + 1) * p.conv_dim + global_row];
        }
        next_history[(p.kernel - 2) * p.conv_dim + global_row] = x_input;

        const float raw = projected_at(projected, inner + p.conv_dim + head, p.projected_dtype)
            + dt_bias[head];
        const float dt = fminf(fmaxf(fmaxf(raw, 0.0f) + log1pf(expf(-fabsf(raw))), p.dt_min), p.dt_max);
        const float decay = expf(dt * a[head]);
        float y = 0.0f;
        const int state_base = global_row * 128;
        #pragma unroll 4
        for (int channel = 0; channel < 128; ++channel) {
            const float updated = decay * state[state_base + channel]
                + (dt * x) * bc[channel];
            next_state[state_base + channel] = updated;
            y += updated * bc[128 + channel];
        }
        y += d[head] * x;
        const float gate = projected_at(projected, global_row, p.projected_dtype);
        gated[local_row] = y * (gate / (1.0f + expf(-gate)));
    }
    __syncthreads();
    float sumsq = 0.0f;
    for (int row = lane; row < group_rows; row += blockDim.x) {
        sumsq += gated[row] * gated[row];
    }
    reduction[lane] = sumsq;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduction[0] / float(group_rows) + p.eps);
    for (int row = lane; row < group_rows; row += blockDim.x) {
        const int global_row = group * group_rows + row;
        store_projected(normalized, global_row, gated[row] * inv_rms * norm_weight[global_row], p.output_dtype);
    }
}
