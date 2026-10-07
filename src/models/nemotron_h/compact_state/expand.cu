typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef signed char int8_t;

struct GpuGroup {
    uint32_t code_offset;
    uint32_t mode;
    float factor;
};

__device__ float half_bits_to_float(uint16_t bits) {
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
        : sign | (uint32_t(exponent + 112) << 23) | (mantissa << 13);
    return __uint_as_float(f32_bits);
}

extern "C" __global__ void expand_compact_mamba(
    const uint8_t* data, const GpuGroup* groups, float* output,
    uint32_t conv_offset, uint32_t conv_words, uint32_t heads,
    uint32_t values, uint32_t channels, uint32_t layers,
    uint32_t layer_words) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t conv_total = layers * conv_words;
    if (tid < conv_total) {
        const uint32_t layer = tid / conv_words;
        const uint32_t word = tid % conv_words;
        const uint32_t offset = conv_offset + tid * 4;
        const uint32_t bits = uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
                              (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
        output[layer * layer_words + word] = __uint_as_float(bits);
        return;
    }
    const uint32_t position = tid - conv_total;
    const uint32_t group_count = layers * heads * channels;
    if (position >= group_count * values) return;
    const uint32_t group_index = position / values;
    const uint32_t value = position % values;
    const uint32_t layer = group_index / (heads * channels);
    const uint32_t head = group_index / channels % heads;
    const uint32_t channel = group_index % channels;
    const GpuGroup group = groups[group_index];
    const uint32_t offset = group.code_offset + value * (1u << group.mode);
    float result;
    if (group.mode == 0) {
        result = float(int8_t(data[offset])) * group.factor;
    } else if (group.mode == 1) {
        const uint16_t bits = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        result = half_bits_to_float(bits) / group.factor;
    } else {
        const uint32_t bits = uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
                              (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
        result = __uint_as_float(bits);
    }
    output[layer * layer_words + conv_words + (head * values + value) * channels + channel] = result;
}
