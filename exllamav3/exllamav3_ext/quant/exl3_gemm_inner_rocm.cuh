#pragma once

// HIP body behind the shared kernel wrappers (exl3_gemm_kernel.cuh,
// exl3_moe_kernel.cuh); dispatcher, kernel tables, autotuner and comp units are
// shared with CUDA. Same contract as the CUDA inner: blockIdx.x slices the
// (k x n) tile space, each column tile assembled by a lock cascade in reverse k
// order (highest-k segment acquires first and accumulates into C in place, the
// k == 0 block writes the final result), no grid-wide sync, and locks used only
// as locks so every caller's disjoint lock range stays disjoint. Decode tiers on
// (bits, cb): 4/6 with mul1 decode lane-local through funnel/byte-dot intrinsics,
// everything else through the shared dq_dispatch. NB: never name a ROCm-side
// header *_hip.cuh - torch's in-place hipify clobbers files with that pattern.

#include "../ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"

// The #undef matters: exl3_moe_common.cuh's 90 KB fallback would exceed
// gfx1100's 64 KB LDS limit.
#undef SMEM_MAX
#define SMEM_MAX (8 * 1024)

// staged in shared memory, declared once per kernel: per-instantiation __shared__
// would request the LDS once per instantiation
#define EXL3_INNER_SH_FLOATS(ts_n) (16 * (ts_n))

// 1: route the 4/6-bit mul1 tensors through the generic k-split lane tier as well
#ifndef EXL3_ROCM_LANE_TIER_ALL
#define EXL3_ROCM_LANE_TIER_ALL 1
#endif

namespace exl3_rocm_inner
{

// funnel shift is native on both vendors
__device__ __forceinline__ uint32_t alignbit16(uint32_t hi, uint32_t lo, int imm)
{
    return __funnelshift_r(lo, hi, imm) & 0xFFFFu;
}

// byte lanes sum to <= 1020, so the low 16 bits are exact
__device__ __forceinline__ uint32_t dot4_add(uint32_t x, uint32_t c)
{
    return __builtin_amdgcn_udot4(x, 0x01010101u, c, false);
}

// same arithmetic as codebook.cuh's decode_mul1_product_2
__device__ __forceinline__ float decode_w_mul1(uint32_t code)
{
    uint32_t u = dot4_add(code * 0x83DCD12Du, 0x6400u) & 0xFFFFu;
    __half h = __ushort_as_half((unsigned short) u);
    __half ki = __ushort_as_half((unsigned short) 0x1EEE);
    __half kb = __ushort_as_half((unsigned short) 0xC931);
    float v = __half2float(h) * __half2float(ki) + __half2float(kb);
    return __half2float(__float2half(v));
}

// mul1 codebook, RDNA fast path. The codebook value is k_inv * (1024 + bytesum(code * M)) + k_bias
// with M = 0x83DCD12D. The 16-bit code times the 32-bit constant mod 2^32 splits into two 16x16
// products (v_mad_u32_u16 / v_mul_lo_u16, full rate, and they read the low half of `code` so the
// upper bits need no masking), and the byte sum is v_sad_u8 against 0 (full rate) rather than the
// quarter-rate v_dot4. With the accumulator 0x6400 the result's low half is the fp16 value
// 1024 + bytesum exactly, which feeds v_fma_mix_f32 directly; the affine correction
// (k_inv, k_bias * sum(x)) is applied once per output instead of once per weight
#ifndef EXL3_ROCM_PIPELINE
#define EXL3_ROCM_PIPELINE 1
#endif
#ifndef EXL3_ROCM_FAST_MUL1
#define EXL3_ROCM_FAST_MUL1 1
#endif
__device__ __forceinline__ uint32_t mul1_h1024(uint32_t code)
{
#if EXL3_ROCM_FAST_MUL1
    // Two full-rate 16-bit multiplies in one asm statement (the compiler folds the C form back
    // into a quarter-rate v_mul_lo_u32 + v_and):
    //   t  = lo16(code) * 0xD12D                    (v_mad_u32_u16, full 32-bit product)
    //   t += (lo16(code) * 0x83DC) << 16            (v_mad_u16 into the high half via op_sel)
    //   == code * 0x83DCD12D mod 2^32; only the low 16 bits of `code` are read
    uint32_t t;
    asm("v_mad_u32_u16 %0, %1, %2, 0\n\tv_mad_u16 %0, %1, %3, %0 op_sel:[0,0,1,1]"
        : "=&v"(t) : "v"(code), "s"(0xD12Du), "s"(0x83DCu));
    return __builtin_amdgcn_sad_u8(t, 0u, 0x6400u);
#else
    const uint32_t p = (code & 0xFFFFu) * 0x83DCD12Du;
    return __builtin_amdgcn_udot4(p, 0x01010101u, 0x6400u, false);
#endif
}
// fp16 constants of the mul1 codebook affine map
#define MUL1_K_INV_BITS  0x1EEEu   //  0.00677 = 1/147.7
#define MUL1_K_BIAS_BITS 0xC931u   // -10.39

// Column lock (ptx.cuh protocol): counts completed k-tiles of one column tile;
// the topmost block resets it to 0 for the next call

__device__ __forceinline__ void lock_acquire(int* lock, int stage)
{
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        unsigned int state;
        do
        {
            state = __hip_atomic_load(a, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
        while (state != (unsigned int) stage);
    }
    __syncthreads();
}

__device__ __forceinline__ void lock_release(int* lock, int val, bool reset)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        if (reset)
        {
            __hip_atomic_store(a, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            return;
        }
        __hip_atomic_fetch_add(a, (unsigned int) val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// phase functions must stay __forceinline__: a __shared__ array whose address
// escapes into a non-inlined callee is demoted to local memory
struct SegCtx
{
    const half* __restrict__ A;
    const uint32_t* __restrict__ B;
    float* __restrict__ sh_c;       // [16][cols] segment partial
    int size_m;
    int size_k;
    int kt0, kt1;                   // trellis k-subtile range of the segment (16 wide each)
    int nsub_total;                 // subtiles per k row of the whole matrix (size_n / 16)
    int subs_tile;                  // subtiles in this column tile (TILESIZE_N / 16)
    int col;                        // column tile index
};

// Tier A: bits 4, cb 2. 32 words per subtile, 8 lanes per (subtile, m row) item

__device__ __forceinline__ void phase1_b4c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;        // global subtile
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 32) + (size_t) n * 32;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define PAIR4(HI, LO, XK)                                                     \
        acc0 += decode_w_mul1(alignbit16(HI, LO, 28)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 24)) * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 20)) * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 16)) * (XK)[9];                      \
        acc1 += decode_w_mul1(alignbit16(HI, LO, 12)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 8))  * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 4))  * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 0))  * (XK)[9];

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 32);
            uint32_t w0 = p32[4 * lane];
            uint32_t w1 = p32[4 * lane + 1];
            uint32_t w2 = p32[4 * lane + 2];
            uint32_t w3 = p32[4 * lane + 3];
            uint32_t prev = p32[(4 * lane + 31) & 31];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            PAIR4(prev, w0, xf)
            PAIR4(w0, w1, xf + 2)
            PAIR4(w1, w2, xf + 4)
            PAIR4(w2, w3, xf + 6)
        }
        #undef PAIR4

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier B: bits 6, cb 2. 48 words per subtile, 6-word window algebra

__device__ __forceinline__ void phase1_b6c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 48) + (size_t) n * 48;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define CA(HI, LO, SH) \
            ((SH) < 32 ? alignbit16(HI, LO, (SH)) : ((HI) >> ((SH) - 32)) & 0xFFFFu)

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 48);
            uint32_t w0 = p32[6 * lane];
            uint32_t w1 = p32[6 * lane + 1];
            uint32_t w2 = p32[6 * lane + 2];
            uint32_t w3 = p32[6 * lane + 3];
            uint32_t w4 = p32[6 * lane + 4];
            uint32_t w5 = p32[6 * lane + 5];
            uint32_t m1 = p32[(6 * lane + 47) % 48];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            acc0 += decode_w_mul1(CA(m1, w0, 26)) * xf[0]
                  + decode_w_mul1(CA(m1, w0, 20)) * xf[1]
                  + decode_w_mul1(CA(m1, w0, 14)) * xf[8]
                  + decode_w_mul1(CA(m1, w0, 8))  * xf[9];
            acc1 += decode_w_mul1(CA(w0, w1, 34)) * xf[0]
                  + decode_w_mul1(CA(w0, w1, 28)) * xf[1]
                  + decode_w_mul1(CA(w0, w1, 22)) * xf[8]
                  + decode_w_mul1(CA(w0, w1, 16)) * xf[9];
            acc0 += decode_w_mul1(CA(w1, w2, 42)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 36)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 30)) * xf[10]
                  + decode_w_mul1(CA(w1, w2, 24)) * xf[11];
            acc1 += decode_w_mul1(CA(w1, w2, 18)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 12)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 6))  * xf[10]
                  + decode_w_mul1(CA(w1, w2, 0))  * xf[11];
            acc0 += decode_w_mul1(CA(w2, w3, 26)) * xf[4]
                  + decode_w_mul1(CA(w2, w3, 20)) * xf[5]
                  + decode_w_mul1(CA(w2, w3, 14)) * xf[12]
                  + decode_w_mul1(CA(w2, w3, 8))  * xf[13];
            acc1 += decode_w_mul1(CA(w3, w4, 34)) * xf[4]
                  + decode_w_mul1(CA(w3, w4, 28)) * xf[5]
                  + decode_w_mul1(CA(w3, w4, 22)) * xf[12]
                  + decode_w_mul1(CA(w3, w4, 16)) * xf[13];
            acc0 += decode_w_mul1(CA(w4, w5, 42)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 36)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 30)) * xf[14]
                  + decode_w_mul1(CA(w4, w5, 24)) * xf[15];
            acc1 += decode_w_mul1(CA(w4, w5, 18)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 12)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 6))  * xf[14]
                  + decode_w_mul1(CA(w4, w5, 0))  * xf[15];
        }
        #undef CA

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier A': generic lane-local decode for any bitrate (integer 1..8 or KA + 0.5, any codebook).
// Same work split as Tier A (8 lanes per (subtile, m row) item, lane L owns the 32 trellis
// positions 32L..32L+31 = output columns L and L+8 over the 16 k rows) but the 16-bit code
// windows are cut with compile-time shifts from the lane's own word run, so no lane reads
// another lane's words and the per-weight cost is a funnel shift plus the codebook decode.
//
// Bit layout (see exl3_dq.cuh): word index ascends with the logical bit index, the MSB of a
// word is its first logical bit, and position e's code is the 16-bit window ending at
// E(e) = sum_{q<=e} bits(q), with bits(q) = bits + (half_k && (q & 1)). A lane's 32 positions
// occupy LANE_BITS = 32 * bits + (half_k ? 16 : 0) logical bits starting at LANE_BITS * L; for
// half_k that start is mid-word on odd lanes, which is fixed by re-aligning the word run 16 bits
// once per k-tile so every window shift below stays a compile-time constant.

template <int bits, bool half_k>
__device__ __forceinline__ constexpr int lane_e_rel(int i)
{
    // end bit (exclusive) of lane position i relative to the lane's aligned stream
    return bits * (i + 1) + (half_k ? ((i + 1) >> 1) : 0);
}

template <int bits, bool half_k, int cb>
__device__ __forceinline__ void phase1_lane(SegCtx& c)
{
    constexpr int SUB_U32 = 8 * bits + (half_k ? 4 : 0);        // words per 16x16 subtile
    constexpr int LANE_BITS = 32 * bits + (half_k ? 16 : 0);    // bits per lane
    constexpr int NW = half_k ? (LANE_BITS + 16 + 31) / 32 : bits;   // words after alignment
    static_assert(NW * 32 >= LANE_BITS, "lane word run too short");

    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    // Decode-shaped calls leave most 8-lane groups idle (items < groups) while each active
    // group walks its whole k segment serially, so the block is latency-bound on one load
    // chain per group. Split the segment across the idle groups instead and reduce the
    // partials through LDS atomics; the segment partial is then a sum of KS chains
    const int cols = c.subs_tile * 16;
    int KS = groups / items;
    if (KS < 1) KS = 1;
    const int seg_len = c.kt1 - c.kt0;
    if (KS > seg_len) KS = seg_len;

    // (barrier first: the previous column tile's write-out may still be reading sh_c)
    __syncthreads();
    for (int i = threadIdx.x; i < c.size_m * cols; i += blockDim.x) c.sh_c[i] = 0.f;
    __syncthreads();

    const int base_bit = LANE_BITS * lane;
    const int bw = base_bit >> 5;
    const bool odd = half_k && ((base_bit & 31) != 0);
    const int bw_prev = (bw == 0) ? (SUB_U32 - 1) : (bw - 1);

    for (int idx = rem; idx < items * KS; idx += groups)
    {
        int item = idx / KS;
        int ks = idx - item * KS;
        int s_ = item % c.subs_tile;
        int n = c.col * c.subs_tile + s_;
        int m = item / c.subs_tile;
        const int t0 = c.kt0 + (int) ((int64_t) seg_len * ks / KS);
        const int t1 = c.kt0 + (int) ((int64_t) seg_len * (ks + 1) / KS);

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * SUB_U32) + (size_t) n * SUB_U32;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f, xsum = 0.f;

        // Software pipeline (depth 1): tile t + 1 is requested before tile t is decoded
        uint32_t bufA[NW + 1], bufB[NW + 1];
        uint32_t xA[8], xB[8];

        auto load_tile = [&](int t, uint32_t* a, uint32_t* xw)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * SUB_U32);
            if constexpr (half_k)
            {
                uint32_t w[NW + 2];
                w[0] = p32[bw_prev];
                #pragma unroll
                for (int k = 0; k < NW; ++k) w[k + 1] = p32[bw + k];
                w[NW + 1] = 0;
                #pragma unroll
                for (int k = 0; k <= NW; ++k)
                    a[k] = odd ? ((w[k] << 16) | (w[k + 1] >> 16)) : w[k];
            }
            else
            {
                a[0] = p32[bw_prev];
                #pragma unroll
                for (int k = 0; k < NW; ++k) a[k + 1] = p32[bw + k];
            }
            const uint4* xp = (const uint4*) (x + t * 16);
            uint4 x0 = xp[0], x1 = xp[1];
            xw[0] = x0.x; xw[1] = x0.y; xw[2] = x0.z; xw[3] = x0.w;
            xw[4] = x1.x; xw[5] = x1.y; xw[6] = x1.z; xw[7] = x1.w;
        };

        auto decode_tile = [&](const uint32_t* a, const uint32_t* xw)
        {

            if constexpr (cb == 2)
            {
                half xh[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    xh[2 * i]     = __ushort_as_half((unsigned short) (xw[i] & 0xFFFFu));
                    xh[2 * i + 1] = __ushort_as_half((unsigned short) (xw[i] >> 16));
                }
                // Explicitly interleaved decode: the per-code chain (alignbit -> mad -> mad -> sad
                // -> fma) is 5 dependent ops with ~4-cycle result latency each, and the compiler's
                // schedule left those latencies exposed (~50% VALU utilisation). Processing 8 codes
                // in lock-step, one stage at a time, keeps 8 independent ops between dependent hops
                // (measured 1.5 -> 2.6 T weights/s in micro/decode_rate9.hip). The inline-asm stages
                // are volatile so the compiler cannot re-serialise them per code
                float b0 = 0.f, b1 = 0.f;
                #pragma unroll
                for (int i0 = 0; i0 < 32; i0 += 8)
                {
                    uint32_t code[8], t[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        const int E = lane_e_rel<bits, half_k>(i0 + j);
                        const int p = (E - 1) >> 5;
                        const int sh = 32 * (p + 1) - E;
                        code[j] = __funnelshift_r(a[p + 1], a[p], sh);
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) asm volatile("v_mad_u32_u16 %0, %1, %2, 0" : "=v"(t[j]) : "v"(code[j]), "s"(0xD12Du));
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) asm volatile("v_mad_u16 %0, %1, %2, %0 op_sel:[0,0,1,1]" : "+v"(t[j]) : "v"(code[j]), "s"(0x83DCu));
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) asm volatile("v_sad_u8 %0, %0, 0, 0x6400" : "+v"(t[j]));
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        const int i = i0 + j;
                        const int g = i >> 3, k = i & 7;
                        const int r = 2 * g + (k & 1) + 8 * ((k >> 1) & 1);
                        const float pr = __half2float(__ushort_as_half((unsigned short) t[j])) * __half2float(xh[r]);
                        if (k < 4) { if (k & 1) b0 += pr; else acc0 += pr; }
                        else       { if (k & 1) b1 += pr; else acc1 += pr; }
                    }
                }
                acc0 += b0; acc1 += b1;
                float sx = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) sx += __half2float(xh[i]);
                xsum += sx;
            }
            else
            {
                float xf[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    xf[2 * i]     = __half2float(__ushort_as_half((unsigned short) (xw[i] & 0xFFFFu)));
                    xf[2 * i + 1] = __half2float(__ushort_as_half((unsigned short) (xw[i] >> 16)));
                }
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                {
                    const int E = lane_e_rel<bits, half_k>(i);
                    const int p = (E - 1) >> 5;
                    const int sh = 32 * (p + 1) - E;
                    uint32_t code = alignbit16(a[p], a[p + 1], sh);
                    float wv = __half2float(decode_3inst<cb>(code));
                    const int g = i >> 3, k = i & 7;
                    const int r = 2 * g + (k & 1) + 8 * ((k >> 1) & 1);
                    if (k < 4) acc0 += wv * xf[r];
                    else       acc1 += wv * xf[r];
                }
            }

        };


#if EXL3_ROCM_PIPELINE
        // even/odd tile split with a fixed trip count: the compiler turned the earlier
        // `for (; t + 1 < t1; t += 2)` form with in-loop conditional loads into a non-terminating
        // loop for the half-integer instances
        const int ntiles = t1 - t0;
        if (ntiles > 0) load_tile(t0, bufA, xA);
        for (int i = 0; i < ntiles; i += 2)
        {
            const bool has_b = (i + 1 < ntiles);
            if (has_b) load_tile(t0 + i + 1, bufB, xB);
            decode_tile(bufA, xA);
            if (i + 2 < ntiles) load_tile(t0 + i + 2, bufA, xA);
            if (has_b) decode_tile(bufB, xB);
        }
#else
        for (int t = t0; t < t1; ++t) { load_tile(t, bufA, xA); decode_tile(bufA, xA); }
#endif

        if constexpr (cb == 2)
        {
            // acc = sum (1024 + s) * x  ->  sum w * x = k_inv * acc + k_bias * sum x
            const float k_inv = __half2float(__ushort_as_half((unsigned short) MUL1_K_INV_BITS));
            const float k_bias = __half2float(__ushort_as_half((unsigned short) MUL1_K_BIAS_BITS));
            acc0 = k_inv * acc0 + k_bias * xsum;
            acc1 = k_inv * acc1 + k_bias * xsum;
        }

        float* out = c.sh_c + (size_t) m * cols + (size_t) s_ * 16;
        if (KS == 1)
        {
            out[lane] = acc0;
            out[lane + 8] = acc1;
        }
        else
        {
            atomicAdd(out + lane, acc0);
            atomicAdd(out + lane + 8, acc1);
        }
    }
}

// Tier C: everything else. One warp per subtile through dq_dispatch

template <int bits, int cb, bool half_k = false>
__device__ __forceinline__ void phase1_dq(SegCtx& c)
{
    // uint32 words per 16x16 subtile: 8 * bits for integer K, 8 * bits + 4 for K + 0.5 (mul1 only)
    constexpr int SUB_U32 = 8 * bits + (half_k ? 4 : 0);

    int lane_id = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    int r0 = 2 * (lane_id & 3);
    int c0 = 2 * (lane_id >> 3) + ((lane_id & 7) >> 2);
    int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };

    for (int idx = warp; idx < c.subs_tile; idx += warps)
    {
        int n = c.col * c.subs_tile + idx;
        const uint32_t* sub_base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * SUB_U32) + (size_t) n * SUB_U32;

        float acc[16][2];
        #pragma unroll
        for (int m = 0; m < 16; ++m) { acc[m][0] = 0.f; acc[m][1] = 0.f; }

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* ptr = sub_base + (size_t) (t - c.kt0) * (c.nsub_total * SUB_U32);

            FragB frag[2];
            dq_dispatch<bits, cb, half_k>(ptr, lane_id * 8, frag[0], frag[1]);

            #pragma unroll
            for (int m = 0; m < 16; ++m)
            {
                if (m < c.size_m)
                {
                    const half* x = c.A + (size_t) m * c.size_k + (size_t) t * 16;
                    float x0 = __half2float(x[rows[0]]);
                    float x1 = __half2float(x[rows[1]]);
                    float x2 = __half2float(x[rows[2]]);
                    float x3 = __half2float(x[rows[3]]);
                    acc[m][0] += __half2float(frag[0][0].x) * x0
                               + __half2float(frag[0][0].y) * x1
                               + __half2float(frag[0][1].x) * x2
                               + __half2float(frag[0][1].y) * x3;
                    acc[m][1] += __half2float(frag[1][0].x) * x0
                               + __half2float(frag[1][0].y) * x1
                               + __half2float(frag[1][1].x) * x2
                               + __half2float(frag[1][1].y) * x3;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < 16; ++m)
        {
            if (m >= c.size_m) continue;
            float a0 = acc[m][0], a1 = acc[m][1];
            a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 2);

            if ((lane_id & 3) == 0)
            {
                float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) idx * 16;
                out[c0] = a0;
                out[c0 + 8] = a1;
            }
        }
    }
}

}  // namespace exl3_rocm_inner

// shared inner entry point; C is [m][n] with row stride size_n. post_scale
// applies only when shmem_out_had is set

template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
__device__ void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int size_n_stride = 0,       // full width of B and C rows when computing a column slice (0: = size_n)
    float* __restrict__ sh = nullptr
)
{
    using namespace exl3_rocm_inner;

    if (size_n_stride == 0) size_n_stride = size_n;
    const int n_full = size_n_stride;    // B rows and C rows span the full matrix width
    constexpr int TS_N = TILESIZE_N;
    float* sh_c = sh;

    int tiles_k = size_k / 16;                 // trellis k-subtiles (16 wide)
    int tiles_n = size_n / TS_N;
    int units = tiles_k * tiles_n;
    int num_slices = gridDim.x;
    int beg = (int) ((int64_t) units * blockIdx.x / num_slices);
    int end = (int) ((int64_t) units * (blockIdx.x + 1) / num_slices);

    SegCtx c;
    c.A = A;
    c.B = (const uint32_t*) B;
    c.sh_c = sh_c;
    c.size_m = size_m;
    c.size_k = size_k;
    c.nsub_total = n_full / 16;
    c.subs_tile = TS_N / 16;

    while (beg < end)
    {
        int col = beg / tiles_k;
        int seg_k0 = beg % tiles_k;
        int seg_k1 = ((end - 1) / tiles_k == col) ? ((end - 1) % tiles_k) + 1 : tiles_k;

        c.kt0 = seg_k0;
        c.kt1 = seg_k1;
        c.col = col;
        int cols = c.subs_tile * 16;

        // Lane-local tiers re-decode the weights once per m row, the warp tier once per
        // subtile: the former wins for decode-shaped m, the latter for prefill chunks
        constexpr int LANE_TIER_MAX_M = 8;
        if constexpr (cb == 2 && bits == 4 && !half_k && !EXL3_ROCM_LANE_TIER_ALL)
        {
            phase1_b4c2(c);
        }
        else if constexpr (cb == 2 && bits == 6 && !half_k && !EXL3_ROCM_LANE_TIER_ALL)
        {
            phase1_b6c2(c);
        }
        else if constexpr (bits > 0)
        {
            if (c.size_m <= LANE_TIER_MAX_M)
                phase1_lane<bits, half_k, cb>(c);
            else
                phase1_dq<bits, cb, half_k>(c);
        }
        // bits == 0 never reaches the inner (the caller switches on K first)

        __syncthreads();

        int lock_i = tiles_k - seg_k1;
        int lock_d = seg_k1 - seg_k0;
        int* lock = &locks[col];
        lock_acquire(lock, lock_i);

        bool first = (lock_i == 0);
        bool last = (lock_i + lock_d == tiles_k);

        if (!first)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    sh_c[i] += ((const float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n];
                else
                    sh_c[i] += __half2float(((const half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n]);
            }
            __syncthreads();
        }

        if (!last)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }
        else if (shmem_out_had)
        {
            // final block: output Hadamard (without post_scale only the
            // 1/sqrt(128) factor, no per-column scale)
            int nb = cols / 128;
            int warp = threadIdx.x >> 5;
            int lane = threadIdx.x & 31;
            int warps = blockDim.x >> 5;
            int slots = size_m * nb;
            for (int slot = warp; slot < slots; slot += warps)
            {
                int m = slot / nb;
                int b = slot % nb;
                const float* p = sh_c + m * cols + b * 128;
                float v0 = p[lane * 4 + 0];
                float v1 = p[lane * 4 + 1];
                float v2 = p[lane * 4 + 2];
                float v3 = p[lane * 4 + 3];

                float s0 = v0 + v1, d0 = v0 - v1;
                float s1 = v2 + v3, d1 = v2 - v3;
                float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

                shuffle_had_f4x32(h0, h1, h2, h3, lane);

                const float rs = 0.088388347648f;   // 1/sqrt(128)
                if (post_scale)
                {
                    const half* sb = post_scale + ((size_t) col * TS_N + b * 128) % n_full;
                    h0 *= rs * __half2float(sb[lane * 4 + 0]);
                    h1 *= rs * __half2float(sb[lane * 4 + 1]);
                    h2 *= rs * __half2float(sb[lane * 4 + 2]);
                    h3 *= rs * __half2float(sb[lane * 4 + 3]);
                }
                else
                {
                    h0 *= rs; h1 *= rs; h2 *= rs; h3 *= rs;
                }

                size_t n0 = (size_t) m * n_full + (size_t) col * TS_N + b * 128 + lane * 4;
                if constexpr (c_fp32)
                {
                    float* out = (float*) C + n0;
                    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3;
                }
                else
                {
                    half* out = (half*) C + n0;
                    out[0] = __float2half(h0); out[1] = __float2half(h1);
                    out[2] = __float2half(h2); out[3] = __float2half(h3);
                }
            }
        }
        else
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }

        lock_release(lock, lock_d, last);

        // phase 1 must store fresh values: the accumulate path adds into sh_c
        beg = (col + 1) * tiles_k;
    }
}
