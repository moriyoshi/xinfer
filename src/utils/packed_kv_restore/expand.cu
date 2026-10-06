typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

struct GpuPage {
  uint64_t destination;
  uint32_t code_offset, param_offset, tail_offset, exception_offset;
  uint32_t exception_count, tokens, heads, channels, old_tokens;
  uint32_t tail_tokens, bits, group_size, slot_offset, key;
};

extern "C" __global__ void expand_dense_packed_kv(
    const GpuPage* pages, const uint8_t* codes, const float* params,
    const uint16_t* tails, const uint32_t* exception_indices,
    const uint16_t* exception_values) {
  const GpuPage page = pages[blockIdx.y];
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t width = page.heads * page.channels;
  if (index >= page.tokens * width) return;
  const uint32_t token = index / width;
  const uint32_t hd = index % width;
  uint16_t word;
  if (token >= page.old_tokens) {
    const uint32_t rank = page.key
        ? hd * page.tail_tokens + token - page.old_tokens
        : (token - page.old_tokens) * width + hd;
    word = tails[page.tail_offset + rank];
  } else {
    const uint32_t rank = page.key
        ? hd * page.old_tokens + token : index;
    const uint32_t bit = rank * page.bits;
    const uint32_t code = (codes[page.code_offset + bit / 8] >> (bit % 8))
        & ((1u << page.bits) - 1u);
    const uint32_t group = page.key
        ? hd * ((page.old_tokens + page.group_size - 1) / page.group_size)
              + token / page.group_size
        : (token * page.heads + hd / page.channels)
              * ((page.channels + page.group_size - 1) / page.group_size)
              + (hd % page.channels) / page.group_size;
    const double value = (double)params[page.param_offset + group * 2]
        + (double)code * (double)params[page.param_offset + group * 2 + 1];
    const uint32_t raw = __float_as_uint((float)value);
    word = (uint16_t)((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
    uint32_t lo = 0, hi = page.exception_count;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      if (exception_indices[page.exception_offset + mid] < index) lo = mid + 1;
      else hi = mid;
    }
    if (lo < page.exception_count &&
        exception_indices[page.exception_offset + lo] == index) {
      word = exception_values[page.exception_offset + lo];
    }
  }
  uint16_t* dst = reinterpret_cast<uint16_t*>(page.destination);
  dst[(page.slot_offset + token) * width + hd] = word;
}
