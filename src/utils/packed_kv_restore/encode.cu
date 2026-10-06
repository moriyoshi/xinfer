typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// Each page occupies a fixed output slot. The host trims the final page and
// the exact tail before building the versioned page envelope.
static __device__ __forceinline__ uint32_t old_tokens(
    uint32_t page, uint32_t tokens, uint32_t exact_from) {
  const uint32_t first = page * 16;
  const uint32_t count = min(16u, tokens - first);
  return min(count, exact_from > first ? exact_from - first : 0u);
}

static __device__ __forceinline__ uint32_t source_index(
    uint32_t page, uint32_t old, uint32_t heads, uint32_t channels,
    uint32_t group, uint32_t element, uint32_t key) {
  const uint32_t width = heads * channels;
  if (key) {
    return (page * 16 + element) * width + group;
  }
  const uint32_t token = group / heads;
  const uint32_t head = group % heads;
  return (page * 16 + token) * width + head * channels + element;
}

extern "C" __global__ void dense_kv_params(
    const uint16_t* input, float* params, uint32_t* invalid,
    uint32_t tokens, uint32_t heads, uint32_t channels,
    uint32_t exact_from, uint32_t bits, uint32_t key,
    uint32_t param_stride) {
  const uint32_t page = blockIdx.y;
  const uint32_t old = old_tokens(page, tokens, exact_from);
  const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t groups = key ? heads * channels : old * heads;
  if (group >= groups || old == 0) return;
  const uint32_t count = key ? old : channels;
  float minimum = __int_as_float(0x7f800000);
  float maximum = __int_as_float(0xff800000);
  for (uint32_t element = 0; element < count; ++element) {
    const uint16_t word = input[source_index(page, old, heads, channels,
                                              group, element, key)];
    const float value = __uint_as_float((uint32_t)word << 16);
    if (!isfinite(value)) {
      atomicOr(invalid, 1u);
      return;
    }
    minimum = fminf(minimum, value);
    maximum = fmaxf(maximum, value);
  }
  const float step = (float)(((double)maximum - (double)minimum)
      / (double)((1u << bits) - 1u));
  params[page * param_stride + group * 2] = minimum;
  params[page * param_stride + group * 2 + 1] = step;
}

extern "C" __global__ void dense_kv_codes(
    const uint16_t* input, const float* params, uint8_t* codes,
    uint32_t tokens, uint32_t heads, uint32_t channels,
    uint32_t exact_from, uint32_t bits, uint32_t key,
    uint32_t param_stride, uint32_t code_stride) {
  const uint32_t page = blockIdx.y;
  const uint32_t old = old_tokens(page, tokens, exact_from);
  const uint32_t width = heads * channels;
  const uint32_t code_count = old * width;
  const uint32_t per_byte = 8 / bits;
  const uint32_t byte = blockIdx.x * blockDim.x + threadIdx.x;
  if (byte >= (code_count + per_byte - 1) / per_byte) return;
  uint8_t packed = 0;
  for (uint32_t lane = 0; lane < per_byte; ++lane) {
    const uint32_t rank = byte * per_byte + lane;
    if (rank >= code_count) break;
    uint32_t group, source;
    if (key) {
      const uint32_t hd = rank / old;
      const uint32_t token = rank % old;
      group = hd;
      source = (page * 16 + token) * width + hd;
    } else {
      group = rank / channels;
      source = page * 16 * width + rank;
    }
    const float minimum = params[page * param_stride + group * 2];
    const float step = params[page * param_stride + group * 2 + 1];
    const float value = __uint_as_float((uint32_t)input[source] << 16);
    const int code = step == 0.0f ? 0 :
        (int)round(((double)value - (double)minimum) / (double)step);
    packed |= (uint8_t)(max(0, min(code, (int)((1u << bits) - 1u)))
                        << (lane * bits));
  }
  codes[page * code_stride + byte] = packed;
}

struct GpuEncodeTile {
  unsigned long long source;
  uint32_t exact_from;
  uint32_t key;
  uint32_t param_offset;
  uint32_t param_stride;
};

extern "C" __global__ void dense_kv_params_batch(
    const GpuEncodeTile* tiles, float* params, uint32_t* invalid,
    uint32_t tokens, uint32_t heads, uint32_t channels,
    uint32_t bits, uint32_t pages) {
  const uint32_t tile = blockIdx.z;
  const GpuEncodeTile desc = tiles[tile];
  const uint16_t* input = reinterpret_cast<const uint16_t*>(desc.source);
  const uint32_t page = blockIdx.y;
  const uint32_t old = old_tokens(page, tokens, desc.exact_from);
  const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t groups = desc.key ? heads * channels : old * heads;
  if (group >= groups || old == 0) return;
  const uint32_t count = desc.key ? old : channels;
  float minimum = __int_as_float(0x7f800000);
  float maximum = __int_as_float(0xff800000);
  for (uint32_t element = 0; element < count; ++element) {
    const uint16_t word = input[source_index(page, old, heads, channels,
                                              group, element, desc.key)];
    const float value = __uint_as_float((uint32_t)word << 16);
    if (!isfinite(value)) {
      atomicOr(invalid, 1u);
      return;
    }
    minimum = fminf(minimum, value);
    maximum = fmaxf(maximum, value);
  }
  const float step = (float)(((double)maximum - (double)minimum)
      / (double)((1u << bits) - 1u));
  const uint32_t offset = desc.param_offset + page * desc.param_stride + group * 2;
  params[offset] = minimum;
  params[offset + 1] = step;
}

extern "C" __global__ void dense_kv_codes_batch(
    const GpuEncodeTile* tiles, const float* params, uint8_t* codes,
    uint32_t tokens, uint32_t heads, uint32_t channels,
    uint32_t bits, uint32_t pages, uint32_t code_stride) {
  const uint32_t tile = blockIdx.z;
  const GpuEncodeTile desc = tiles[tile];
  const uint16_t* input = reinterpret_cast<const uint16_t*>(desc.source);
  const uint32_t page = blockIdx.y;
  const uint32_t old = old_tokens(page, tokens, desc.exact_from);
  const uint32_t width = heads * channels;
  const uint32_t code_count = old * width;
  const uint32_t per_byte = 8 / bits;
  const uint32_t byte = blockIdx.x * blockDim.x + threadIdx.x;
  if (byte >= (code_count + per_byte - 1) / per_byte) return;
  uint8_t packed = 0;
  for (uint32_t lane = 0; lane < per_byte; ++lane) {
    const uint32_t rank = byte * per_byte + lane;
    if (rank >= code_count) break;
    uint32_t group, source;
    if (desc.key) {
      const uint32_t hd = rank / old;
      const uint32_t token = rank % old;
      group = hd;
      source = (page * 16 + token) * width + hd;
    } else {
      group = rank / channels;
      source = page * 16 * width + rank;
    }
    const uint32_t param = desc.param_offset + page * desc.param_stride + group * 2;
    const float minimum = params[param];
    const float step = params[param + 1];
    const float value = __uint_as_float((uint32_t)input[source] << 16);
    const int code = step == 0.0f ? 0 :
        (int)round(((double)value - (double)minimum) / (double)step);
    packed |= (uint8_t)(max(0, min(code, (int)((1u << bits) - 1u)))
                        << (lane * bits));
  }
  codes[(tile * pages + page) * code_stride + byte] = packed;
}

extern "C" __global__ void dense_kv_tail_batch(
    const GpuEncodeTile* tiles, uint16_t* tails,
    uint32_t tokens, uint32_t width, uint32_t tail_stride) {
  const uint32_t tile = blockIdx.y;
  const GpuEncodeTile desc = tiles[tile];
  const uint16_t* input = reinterpret_cast<const uint16_t*>(desc.source);
  const uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= (tokens - desc.exact_from) * width) return;
  tails[tile * tail_stride + rank] = input[desc.exact_from * width + rank];
}
