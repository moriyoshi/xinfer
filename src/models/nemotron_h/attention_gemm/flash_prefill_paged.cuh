/**
 * @brief Native Flash Attention v2 — paged prefill with BF16 KV cache, SM80+.
 *
 * This CUDA kernel is developed for xInfer (vLLM.rs) project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/flash/flash_prefill_paged.cuh
 *
 * Copyright (c) 2026, Guoqing Bao.  All rights reserved.
 *
 * @details
 * Tiled prefill kernel with 2-stage pipelining (async shared memory loads).
 * Processes Q/K/V tiles in shared memory with BF16→float dequantization.
 * Supports sliding window attention and softcapping. Adapted from atlas
 * inferspark kernels for high-bandwidth GPUs (A100/H100/H800/B200).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Xinfer's Nemotron-H specialization expands BR from 32 to 64 with eight
// warps. Each K/V tile is reused across twice as many query rows while the
// per-warp QK and PV operation order remains the same.
// Key differences from atlas GB10 (SM121):
//   - Uses ldmatrix.x4 / ldmatrix.trans for efficient smem→register loads (SM80+)
//   - Uses IEEE expf for long-context softmax stability (was __expf)
//   - cp.async.cg for vectorized 16-byte global→shared loads
//
// Algorithm: Tiled online softmax over KV sequence (Flash Attention v2).
//   For each Q tile: iterate over KV tiles via block_table:
//     1. S = Q @ K^T * scale        (mma.sync m16n8k16)
//     2. Optional softcap: S = softcap * tanh(S / softcap)
//     3. Causal + sliding window mask
//     4. Online softmax in registers (warp-shuffle max/sum)
//     5. O += P @ V                 (mma.sync m16n8k16)
//
// Tile sizes: BR=64, BC=32 (8 warps, 256 threads)
// Q/O layout: [q_len, num_q_heads, head_dim] BF16
// KV cache:   [num_blocks, block_size, num_kv_heads, head_dim] BF16 (NHD paged)
//
// This copy is specialized for FLASH_HDIM=128.

#include "flash_sm_compat.cuh"

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#if FLASH_HDIM != 128
#error "Nemotron-H attention GEMM supports head width 128 only"
#endif

#define BR 64
#define BC 32

#if FLASH_HDIM <= 256
#define PAD_KV 8
#else
#define PAD_KV 0
#endif

#define HDIM FLASH_HDIM
#define HDIM_PAD (HDIM + PAD_KV)
#define PAD_P 8
#define N_TILES_PER_WARP ((HDIM / 8) / 2)
#define TILE_CHUNKS (BC * (HDIM / 8))
#define NUM_THREADS 256

// Paged KV tile loader: cp.async from scattered pages to contiguous smem.
// block_size is always power-of-2 in practice; __ffs gives bit position for shift.
#define LOAD_KV_TILE_BF16(cache, direct, direct_current, seq_start, q_offset, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const unsigned int _cpr = HDIM / 8; \
        const unsigned long long _ps = (unsigned long long)cache_block_size * num_kv_heads * head_dim; \
        const unsigned long long _rs = (unsigned long long)num_kv_heads * head_dim; \
        const unsigned int _bs_shift = __ffs(cache_block_size) - 1; \
        const unsigned int _bs_mask  = cache_block_size - 1; \
        for (unsigned int _i = (t); _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            unsigned int _sa = __cvta_generic_to_shared(&(smem)[_row * HDIM_PAD + _col]); \
            if (_pos < (kv_l)) { \
                const void* _gm; \
                if (direct_current && _pos >= (q_offset)) { \
                    _gm = (const void*)((direct) + ((unsigned long long)(seq_start) + _pos - (q_offset)) * _rs \
                        + (unsigned long long)(kvh) * head_dim + _col); \
                } else { \
                    unsigned int _lb = _pos >> _bs_shift; \
                    unsigned int _bo = _pos & _bs_mask; \
                    unsigned int _pb = __ldg(&(bt)[_lb]); \
                    _gm = (const void*)((cache) + (unsigned long long)_pb * _ps + (unsigned long long)_bo * _rs \
                        + (unsigned long long)(kvh) * head_dim + _col); \
                } \
                FLASH_CP_ASYNC(_sa, _gm); \
            } else { \
                *((uint4*)&(smem)[_row * HDIM_PAD + _col]) = make_uint4(0,0,0,0); \
            } \
        } \
    } while(0)

// ============================================================================
// BR=64 prefill kernel (8 warps, 256 threads)
//
// Warp roles:
//   QK^T: 8 warps — four 16-row groups, two N halves
//         N-split: warps 0-3 use columns 0-15; warps 4-7 use 16-31
//   PV:   all 8 warps — four 16-row groups, two output-column halves
//         Each warp handles N_TILES_PER_WARP N-tiles
//
// Shared memory:
//   Q:   [64][HDIM_PAD] BF16
//   K:   [2][32][HDIM_PAD] BF16  (double-buffered)
//   V:   [32][HDIM_PAD] BF16
//   P:   [64][40] BF16
//   m/l: [64][2] FP32
// ============================================================================

#if HDIM <= 256

#define PREFILL_USE_DYNAMIC_SMEM 1

template<typename HalfT>
__global__ void
#if PREFILL_USE_DYNAMIC_SMEM
__launch_bounds__(NUM_THREADS)
#endif
flash_prefill_paged(
    const HalfT* __restrict__ Q,
    const HalfT* __restrict__ K_cache,
    const HalfT* __restrict__ V_cache,
    const HalfT* __restrict__ K_direct,
    const HalfT* __restrict__ V_direct,
    const bool direct_current,
    HalfT* __restrict__ O,
    const int* __restrict__ block_tables,
    const unsigned int block_table_stride,
    const unsigned int* __restrict__ cu_seqlens_q,
    const unsigned int* __restrict__ context_lens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_block_size,
    const unsigned int sliding_window,
    const unsigned int causal,
    const float inv_sqrt_d,
    const float softcap
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int seq_idx = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    if (q_head >= num_q_heads) return;

    const unsigned int q_seq_start = cu_seqlens_q[seq_idx];
    const unsigned int q_len = cu_seqlens_q[seq_idx + 1] - q_seq_start;
    const unsigned int kv_len = context_lens[seq_idx];
    const unsigned int q_offset = kv_len > q_len ? kv_len - q_len : 0;
    const int* block_table = block_tables + seq_idx * block_table_stride;

    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_tile_end = min(q_start + BR, q_len);
    const unsigned int q_tile_len = q_tile_end - q_start;
    const unsigned int q_seq_stride = num_q_heads * head_dim;
    const unsigned int kv_head = q_head / (num_q_heads / num_kv_heads);

    Q += q_seq_start * q_seq_stride;
    O += q_seq_start * q_seq_stride;

#if PREFILL_USE_DYNAMIC_SMEM
    extern __shared__ __align__(16) unsigned char prefill_smem_dyn[];
    HalfT* smem_Q = reinterpret_cast<HalfT*>(prefill_smem_dyn);
    HalfT* smem_K_flat = smem_Q + BR * HDIM_PAD;
    HalfT* smem_V = smem_K_flat + 2 * BC * HDIM_PAD;
    HalfT* smem_P = smem_V + BC * HDIM_PAD;
    float* smem_ml = reinterpret_cast<float*>(smem_P + BR * (BC + PAD_P));
    #define SMEM_K_PAGED(buf, idx) smem_K_flat[(buf) * BC * HDIM_PAD + (idx)]
#else
    __shared__ HalfT smem_Q[BR * HDIM_PAD];
    __shared__ HalfT smem_K_arr[2][BC * HDIM_PAD];
    __shared__ HalfT smem_V[BC * HDIM_PAD];
    __shared__ HalfT smem_P[BR * (BC + PAD_P)];
    __shared__ float smem_ml[BR * 2];
    #define SMEM_K_PAGED(buf, idx) smem_K_arr[buf][idx]
#endif

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m = (warp_id & 3) * 16;
    const unsigned int qk_n_start = (warp_id >> 2) * 2;
    const unsigned int pv_warp_m = (warp_id & 3) * 16;
    const unsigned int pv_n_start = (warp_id >> 2) * N_TILES_PER_WARP;
    const unsigned int p_stride = BC + PAD_P;

    float acc_o[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP; i++) {
        acc_o[i][0] = 0.f; acc_o[i][1] = 0.f;
        acc_o[i][2] = 0.f; acc_o[i][3] = 0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f, l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + BC - 1) / BC;
    unsigned int kv_block_start = 0;
    if (causal) {
        unsigned int mx = (q_offset + q_tile_end - 1) / BC;
        num_kv_blocks = min(num_kv_blocks, mx + 1);
    }
    if (sliding_window > 0) {
        unsigned int earliest_q = q_offset + q_start;
        unsigned int earliest_visible = (earliest_q >= sliding_window) ? (earliest_q - sliding_window + 1) : 0u;
        kv_block_start = earliest_visible / BC;
    }

    // Load Q + K[0] + K[1] (prefetch two K tiles for pipeline)
    {
        const unsigned int cpr = HDIM / 8;
        for (unsigned int idx = tid; idx < (BR * (HDIM / 8)); idx += NUM_THREADS) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * HDIM_PAD + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start + row) * q_seq_stride + q_head * head_dim + col];
                FLASH_CP_ASYNC(sa, gm);
            } else {
                *((uint4*)&smem_Q[row * HDIM_PAD + col]) = make_uint4(0, 0, 0, 0);
            }
        }
        if (kv_block_start < num_kv_blocks) {
            LOAD_KV_TILE_BF16(K_cache, K_direct, direct_current, q_seq_start, q_offset, block_table, (&SMEM_K_PAGED(0, 0)), kv_block_start * BC, kv_len, kv_head, tid, NUM_THREADS);
        }
        FLASH_ASYNC_COMMIT(); // group 0: Q + K[0]
        if (kv_block_start + 1 < num_kv_blocks) {
            LOAD_KV_TILE_BF16(K_cache, K_direct, direct_current, q_seq_start, q_offset, block_table, (&SMEM_K_PAGED(1, 0)), (kv_block_start + 1) * BC, kv_len, kv_head, tid, NUM_THREADS);
            FLASH_ASYNC_COMMIT(); // group 1: K[1]
        }
        FLASH_ASYNC_WAIT();
    }
    __syncthreads();

    for (unsigned int kv_block = kv_block_start; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC;
        unsigned int kv_end = min(kv_start + BC, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;
        unsigned int buf = kv_block & 1;

        // Async V load (overlaps with QK^T)
        LOAD_KV_TILE_BF16(V_cache, V_direct, direct_current, q_seq_start, q_offset, block_table, smem_V, kv_start, kv_len, kv_head, tid, NUM_THREADS);
        FLASH_ASYNC_COMMIT();

        // QK^T: eight warps, four M groups and two N halves
        float acc_s[2][4];
        {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                acc_s[i][0] = 0.f; acc_s[i][1] = 0.f;
                acc_s[i][2] = 0.f; acc_s[i][3] = 0.f;
            }

            const unsigned int* sQ32 = (const unsigned int*)smem_Q;
            const unsigned int* sK32 = (const unsigned int*)(&SMEM_K_PAGED(buf, 0));
            const unsigned int hdim_pad_u32 = HDIM_PAD / 2;

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM / 16); ks++) {
                unsigned int kb_u32 = ks * 8;
                unsigned int ar0 = qk_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int aq_off = tid_in_group + kb_u32;

                unsigned int a0 = sQ32[ar0 * hdim_pad_u32 + aq_off];
                unsigned int a1 = sQ32[ar1 * hdim_pad_u32 + aq_off];
                unsigned int a2 = sQ32[ar0 * hdim_pad_u32 + aq_off + 4];
                unsigned int a3 = sQ32[ar1 * hdim_pad_u32 + aq_off + 4];

                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    unsigned int nc = (qk_n_start + nt) * 8 + group_id;
                    unsigned int bk_off = tid_in_group + kb_u32;
                    unsigned int b0 = sK32[nc * hdim_pad_u32 + bk_off];
                    unsigned int b1 = sK32[nc * hdim_pad_u32 + bk_off + 4];

                    FLASH_MMA_K16(acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_s[nt][0], acc_s[nt][1], acc_s[nt][2], acc_s[nt][3]);
                }
            }
        }

        // Scale, softcap, mask (all 8 warps, each on its 2 N-tiles)
        unsigned int row0 = qk_warp_m + group_id, row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
            acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;

            if (softcap > 0.f) {
                acc_s[nt][0] = softcap * tanhf(acc_s[nt][0] / softcap);
                acc_s[nt][1] = softcap * tanhf(acc_s[nt][1] / softcap);
                acc_s[nt][2] = softcap * tanhf(acc_s[nt][2] / softcap);
                acc_s[nt][3] = softcap * tanhf(acc_s[nt][3] / softcap);
            }

            unsigned int c0 = (qk_n_start + nt) * 8 + tid_in_group * 2, c1 = c0 + 1;
            unsigned int qr0 = q_offset + q_start + row0, qr1 = q_offset + q_start + row1;

            if (causal) {
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
            }
            if (sliding_window > 0) {
                if (qr0 >= kv_start + c0 && qr0 - (kv_start + c0) >= sliding_window) acc_s[nt][0] = -1e30f;
                if (qr0 >= kv_start + c1 && qr0 - (kv_start + c1) >= sliding_window) acc_s[nt][1] = -1e30f;
                if (qr1 >= kv_start + c0 && qr1 - (kv_start + c0) >= sliding_window) acc_s[nt][2] = -1e30f;
                if (qr1 >= kv_start + c1 && qr1 - (kv_start + c1) >= sliding_window) acc_s[nt][3] = -1e30f;
            }
            if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
            if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
            if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
            if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
        }

        // Partial row-max from this warp's 2 N-tiles
        float rmax0 = fmaxf(fmaxf(acc_s[0][0], acc_s[0][1]), fmaxf(acc_s[1][0], acc_s[1][1]));
        float rmax1 = fmaxf(fmaxf(acc_s[0][2], acc_s[0][3]), fmaxf(acc_s[1][2], acc_s[1][3]));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

        // Cross-warp max: warps 0-3 and 4-7 share corresponding row groups
        // smem_ml[row * 2 + half] where half = warp_id >> 2
        if (tid_in_group == 0) {
            smem_ml[row0 * 2 + (warp_id >> 2)] = rmax0;
            smem_ml[row1 * 2 + (warp_id >> 2)] = rmax1;
        }
        __syncthreads();

        // Combine full row max from both N-halves
        {
            unsigned int half = warp_id >> 2;
            float other0 = smem_ml[row0 * 2 + (1 - half)];
            float other1 = smem_ml[row1 * 2 + (1 - half)];
            rmax0 = fmaxf(rmax0, other0);
            rmax1 = fmaxf(rmax1, other1);
        }

        // Online softmax rescale
        float mn0 = fmaxf(m_r0, rmax0);
        if (mn0 != m_r0) {
            float eo0 = expf(m_r0 - mn0); l_r0 *= eo0;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
            m_r0 = mn0;
        }
        float mn1 = fmaxf(m_r1, rmax1);
        if (mn1 != m_r1) {
            float eo1 = expf(m_r1 - mn1); l_r1 *= eo1;
            #pragma unroll
            for (int i = 0; i < N_TILES_PER_WARP; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
            m_r1 = mn1;
        }

        // exp(s - max), write P to smem, partial l (all 8 warps)
        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            float p00 = expf(acc_s[nt][0] - m_r0), p01 = expf(acc_s[nt][1] - m_r0);
            float p10 = expf(acc_s[nt][2] - m_r1), p11 = expf(acc_s[nt][3] - m_r1);
            sum0 += p00 + p01; sum1 += p10 + p11;
            unsigned int c0 = (qk_n_start + nt) * 8 + tid_in_group * 2;
            smem_P[row0 * p_stride + c0]     = FLASH_FROM_FLOAT(p00);
            smem_P[row0 * p_stride + c0 + 1] = FLASH_FROM_FLOAT(p01);
            smem_P[row1 * p_stride + c0]     = FLASH_FROM_FLOAT(p10);
            smem_P[row1 * p_stride + c0 + 1] = FLASH_FROM_FLOAT(p11);
        }
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
        sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);

        // Cross-warp l sum: accumulate partial sums from both N-halves
        if (tid_in_group == 0) {
            smem_ml[row0 * 2 + (warp_id >> 2)] = sum0;
            smem_ml[row1 * 2 + (warp_id >> 2)] = sum1;
        }
        __syncthreads();
        {
            l_r0 += smem_ml[row0 * 2] + smem_ml[row0 * 2 + 1];
            l_r1 += smem_ml[row1 * 2] + smem_ml[row1 * 2 + 1];
        }

        // Wait for V
        FLASH_ASYNC_WAIT();
        __syncthreads();

        // Preload K[i+2] into the next-next buffer (overlaps with PV compute)
        if (kv_block + 2 < num_kv_blocks) {
            LOAD_KV_TILE_BF16(K_cache, K_direct, direct_current, q_seq_start, q_offset, block_table, (&SMEM_K_PAGED(kv_block & 1, 0)),
                (kv_block + 2) * BC, kv_len, kv_head, tid, NUM_THREADS);
            FLASH_ASYNC_COMMIT();
        }

        // PV MMA (all 8 warps)
        {
            const unsigned int* sP32 = (const unsigned int*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_V;
            const unsigned int p_stride_u32 = p_stride / 2;
            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int pk_off = ks * 8 + tid_in_group;
                unsigned int ar0 = pv_warp_m + group_id, ar1 = ar0 + 8;
                unsigned int a0 = sP32[ar0 * p_stride_u32 + pk_off];
                unsigned int a1 = sP32[ar1 * p_stride_u32 + pk_off];
                unsigned int a2 = sP32[ar0 * p_stride_u32 + pk_off + 4];
                unsigned int a3 = sP32[ar1 * p_stride_u32 + pk_off + 4];
                #pragma unroll
                for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
                    unsigned int nc = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = ks * 16 + tid_in_group * 2, k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sV[(k0 + 1) * HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k0 * HDIM_PAD + nc];
                    unsigned int b1 = ((unsigned int)sV[(k1 + 1) * HDIM_PAD + nc] << 16) |
                                      (unsigned int)sV[k1 * HDIM_PAD + nc];
                    FLASH_MMA_K16(acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc_o[nt][0], acc_o[nt][1], acc_o[nt][2], acc_o[nt][3]);
                }
            }
        }

        if (kv_block + 1 < num_kv_blocks) {
            FLASH_ASYNC_WAIT();
        }
        __syncthreads();
    }

    // Final normalization and store
    {
        unsigned int r0 = pv_warp_m + group_id, r1 = r0 + 8;
        float il0 = (l_r0 > 0.f) ? (1.f / l_r0) : 0.f;
        float il1 = (l_r1 > 0.f) ? (1.f / l_r1) : 0.f;

        HalfT* ob = O + q_head * head_dim;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
            unsigned int c0 = (pv_n_start + nt) * 8 + tid_in_group * 2;
            unsigned int gr0 = q_start + r0, gr1 = q_start + r1;
            if (gr0 < q_len && r0 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(acc_o[nt][0] * il0));
                unsigned int hi = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(acc_o[nt][1] * il0));
                *(unsigned int*)&ob[gr0 * q_seq_stride + c0] = lo | (hi << 16);
            }
            if (gr1 < q_len && r1 < q_tile_len && c0 < head_dim) {
                unsigned int lo = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(acc_o[nt][2] * il1));
                unsigned int hi = (unsigned int)FLASH_AS_USHORT(FLASH_FROM_FLOAT(acc_o[nt][3] * il1));
                *(unsigned int*)&ob[gr1 * q_seq_stride + c0] = lo | (hi << 16);
            }
        }
    }
}
#undef SMEM_K_PAGED
#undef PREFILL_USE_DYNAMIC_SMEM

#endif // HDIM <= 256
