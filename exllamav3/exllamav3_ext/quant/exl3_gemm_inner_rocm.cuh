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
// Fused input Hadamard (EXL3_ROCM_FUSED_HAD): each block rotates the x slice of its k segment
// into LDS instead of a separate Hadamard kernel writing A_had to global memory. Segment length
// is bounded by the autotuned grid (<= ~230 k tiles for the shapes in use); larger segments fall
// back to a per-tile in-register rotation path... no: they are split by the host (see
// exl3_gemm.cu), so the LDS slice always suffices. Budget: EXL3_X_LDS_TILES k tiles per m row
#ifndef EXL3_ROCM_FUSED_HAD
#define EXL3_ROCM_FUSED_HAD 1
#endif
#ifndef EXL3_X_LDS_TILES
#define EXL3_X_LDS_TILES 256                                 // 4096 k: 8 KB LDS; 128 forced min grids of 100-400 blocks on the 5120-k shapes (measured: 256 +1 tok/s, 384/512 lose occupancy)
#endif
#define EXL3_X_LDS_FLOATS (EXL3_X_LDS_TILES * 16 / 2)        // fp16 halves stored as floats/2
// 1: deterministic in-block k-split reduction (slots + ordered sum) instead of LDS atomicAdd
#ifndef EXL3_ROCM_DET_KSPLIT
#define EXL3_ROCM_DET_KSPLIT 1
#endif
// 1: ordered-partials epilogue (no inter-block waiting) for size_m <= 16; 0: column locks
#ifndef EXL3_ROCM_ORDERED_EPILOGUE
#define EXL3_ROCM_ORDERED_EPILOGUE 1
#endif
// k-split slots (EXL3_ROCM_DET_KSPLIT): one 16-float slot per 8-lane group, max 512 threads = 64 groups
#define EXL3_KSLOT_FLOATS (64 * 16)
#define EXL3_INNER_SH_FLOATS(ts_n) (16 * (ts_n) + (EXL3_ROCM_FUSED_HAD ? EXL3_X_LDS_FLOATS : 0) + ((EXL3_ROCM_DET_KSPLIT || EXL3_ROCM_ORDERED_EPILOGUE) ? EXL3_KSLOT_FLOATS : 0))

// 1: route the 4/6-bit mul1 tensors through the generic k-split lane tier as well
#ifndef EXL3_ROCM_LANE_TIER_ALL
#define EXL3_ROCM_LANE_TIER_ALL 1
#endif

#ifdef EXL3_ROCM_PROBE
// gfx11: s_sendmsg_rtn_b64 REALTIME (100 MHz constant clock, consistent across CUs)
__device__ __forceinline__ unsigned long long __probe_now()
{
    unsigned long long t;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_waitcnt lgkmcnt(0)" : "=s"(t));
    return t;
}
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
#ifndef EXL3_ROCM_SLICE_PERMUTE
#define EXL3_ROCM_SLICE_PERMUTE 1
#endif
#ifndef EXL3_ROCM_PAIRED_DOT2
#define EXL3_ROCM_PAIRED_DOT2 1    // sad_hi_u8 + dot2_f32_f16 epilogue (see phase1_lane)
#endif
// 1: one vector load per lane per tile (+ shuffle for the preceding word) instead of NW+1 scalar loads
#ifndef EXL3_ROCM_VEC_LOAD
#define EXL3_ROCM_VEC_LOAD 0   // measured: b96 + shuffle is SLOWER than 4 scalar dword loads (38.7 -> 37.8 tok/s; the shuffle sits on the critical path)
#endif
#ifndef EXL3_ROCM_ILV
#define EXL3_ROCM_ILV 8     // codes decoded in lock-step per stage (see phase1_lane)
#endif
// software pipeline of the lane tier's k loop:
//   1: depth 1 (load t+1 before decode t)               2 buffers of (weights + x)
//   2: depth 2 (load t+2 before decode t)               3 buffers of (weights + x)
//   3: depth 2 for the weights, x just-in-time          3 weight buffers, 1 x buffer
//   4: two independent half-segment chains, depth 1     4 weight buffers, 1 x buffer, 2 acc sets
//   5: depth 1, x just-in-time                         (86 VGPRs; slower: ds_load wait lands before the decode)
//   6: depth 2 via a 2-buffer ring, x jit, 2 acc sets    (91 VGPRs, 4 blocks/CU; = default within noise)
//   7: as 6 with one acc set                            (87 VGPRs; slower)
//   8/9: diagnostics (default loop in alternative forms)
// Measured 2026-09-27 on the lean decode kernel: 1 = 41.3 tok/s, 6 = 41.1, 5 = 40.7, 7 = 40.5, 3 = 40.4
#ifndef EXL3_ROCM_PIPELINE
#define EXL3_ROCM_PIPELINE 1
#endif
// 1: sched_barrier between the prefetch and the decode in the depth-1 loop
#ifndef EXL3_ROCM_SCHED_BARRIER
#define EXL3_ROCM_SCHED_BARRIER 0
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
#ifdef EXL3_ROCM_BOUNDS
        // debug: the column lock relies on every block of the grid being co-resident (a block
        // spins on the blocks above it in k). If the grid exceeds the real occupancy the GPU
        // deadlocks and takes the machine with it -> bounded spin, trap instead
        unsigned int spins = 0;
#endif
        do
        {
            state = __hip_atomic_load(a, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#ifdef EXL3_ROCM_BOUNDS
            if (++spins > 200000000u) __builtin_trap();   // ~ seconds
#endif
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
    const half* __restrict__ xl;    // fused-had: rotated x slice in LDS, [m][ (kt1-kt0)*16 ] (nullptr: read A)
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
#if EXL3_ROCM_DET_KSPLIT
    // deterministic k-split reduction: each (item, ks) chain writes its 16 outputs to its own LDS
    // slot (KS * items = groups slots, region after sh_c and the x slice), summed in ks order after
    float* kslots = c.sh_c + 16 * c.subs_tile * 16 + (EXL3_ROCM_FUSED_HAD ? EXL3_X_LDS_FLOATS : 0);
    if (KS == 1)
        for (int i = threadIdx.x; i < c.size_m * cols; i += blockDim.x) c.sh_c[i] = 0.f;
#else
    for (int i = threadIdx.x; i < c.size_m * cols; i += blockDim.x) c.sh_c[i] = 0.f;
#endif
    __syncthreads();

    const int base_bit = LANE_BITS * lane;
    const int bw = base_bit >> 5;
    const bool odd = half_k && ((base_bit & 31) != 0);
    const int bw_prev = (bw == 0) ? (SUB_U32 - 1) : (bw - 1);

    for (int idx = rem; idx < items * KS; idx += groups)
    {
        // adjacent groups of a wave take adjacent subtiles of the same k range (wave-coherent
        // trellis rows and a wave-uniform x slice: measured ~7% faster than adjacent k-splits)
        int ks = idx / items;
        int item = idx - ks * items;
        int s_ = item % c.subs_tile;
        int n = c.col * c.subs_tile + s_;
        int m = item / c.subs_tile;
        const int t0 = c.kt0 + (int) ((int64_t) seg_len * ks / KS);
        const int t1 = c.kt0 + (int) ((int64_t) seg_len * (ks + 1) / KS);

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * SUB_U32) + (size_t) n * SUB_U32;
        // x: LDS slice (fused had, indexed from the segment start) or the global rotated A
        const int seg_k = (c.kt1 - c.kt0) * 16;
        // x row base and the tile index offset (LDS slice is indexed from the segment start)
        const half* x = c.xl ? (c.xl + (size_t) m * seg_k) : (c.A + (size_t) m * c.size_k);
        const int x_t0 = c.xl ? c.kt0 : 0;
        float acc0 = 0.f, acc1 = 0.f, xsum = 0.f;

        // Software pipeline (depth 1): tile t + 1 is requested before tile t is decoded
        uint32_t bufA[NW + 1], bufB[NW + 1];
        uint32_t xA[8], xB[8];

        // weight words only (x fetched separately: variant 3 = weights 2 ahead, x just-in-time)
        auto load_w = [&](int t, uint32_t* a)
        {
#ifdef EXL3_ROCM_BOUNDS
            if (t < c.kt0 || t >= c.kt1 || t < t0 || t >= t1 || t * 16 >= c.size_k) __builtin_trap();
#endif
#ifdef EXL3_ROCM_EXPERIMENT_FAKE_LOAD
            // experiment: no global weight loads (VALU ceiling); words derived from t and the lane
            const bool fake = true;
#else
            const bool fake = false;
#endif
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * SUB_U32);
            if (fake)
            {
                #pragma unroll
                for (int k = 0; k <= NW; ++k) a[k] = (uint32_t) (t * 2654435761u) ^ (lane * 40503u + k * 0x9E3779B9u);
            }
            else if constexpr (half_k)
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
#if EXL3_ROCM_VEC_LOAD
                // integer K: the lane's NW words are contiguous and NW*4-byte aligned -> one vector
                // load (b64 / b96 / b128); the preceding word comes from the neighbouring lane
                // (lane 0: lane 7's last word = the subtile's last word) via a wave shuffle
                if constexpr (NW == 3)
                {
                    typedef __attribute__((address_space(1))) const uint3 gl_u3;
                    gl_u3* vp = (gl_u3*) (p32 + bw);
                    uint3 v; v.x = vp->x; v.y = vp->y; v.z = vp->z;
                    a[1] = v.x; a[2] = v.y; a[3] = v.z;
                    a[0] = __shfl(v.z, (lane + 7) & 7, 8);
                }
                else if constexpr (NW == 4)
                {
                    typedef __attribute__((address_space(1))) const uint4 gl_u4;
                    gl_u4* vp = (gl_u4*) (p32 + bw);
                    uint4 v; v.x = vp->x; v.y = vp->y; v.z = vp->z; v.w = vp->w;
                    a[1] = v.x; a[2] = v.y; a[3] = v.z; a[4] = v.w;
                    a[0] = __shfl(v.w, (lane + 7) & 7, 8);
                }
                else if constexpr (NW == 2)
                {
                    typedef __attribute__((address_space(1))) const uint2 gl_u2;
                    gl_u2* vp = (gl_u2*) (p32 + bw);
                    uint2 v; v.x = vp->x; v.y = vp->y;
                    a[1] = v.x; a[2] = v.y;
                    a[0] = __shfl(v.y, (lane + 7) & 7, 8);
                }
                else
#endif
                {
                    a[0] = p32[bw_prev];
                    #pragma unroll
                    for (int k = 0; k < NW; ++k) a[k + 1] = p32[bw + k];
                }
            }
        };
        auto load_x = [&](int t, uint32_t* xw)
        {
            if (c.xl)
            {
                typedef __attribute__((address_space(3))) const uint32_t lds_u32;
                lds_u32* xp = (lds_u32*) (const uint32_t*) (x + (t - x_t0) * 16);
                #pragma unroll
                for (int k = 0; k < 8; ++k) xw[k] = xp[k];
                return;
            }
            typedef __attribute__((address_space(1))) const uint4 gl_uint4;
            gl_uint4* xp = (gl_uint4*) (const uint4*) (x + t * 16);
            uint4 x0, x1;
            x0.x = xp[0].x; x0.y = xp[0].y; x0.z = xp[0].z; x0.w = xp[0].w;
            x1.x = xp[1].x; x1.y = xp[1].y; x1.z = xp[1].z; x1.w = xp[1].w;
            xw[0] = x0.x; xw[1] = x0.y; xw[2] = x0.z; xw[3] = x0.w;
            xw[4] = x1.x; xw[5] = x1.y; xw[6] = x1.z; xw[7] = x1.w;
        };

        auto load_tile = [&](int t, uint32_t* a, uint32_t* xw)
        {
#ifdef EXL3_ROCM_BOUNDS
            // debug: a tile index outside the segment means a pipeline bug -> trap (queue error,
            // process abort) instead of an unmapped read that hangs the GPU / the whole machine
            if (t < c.kt0 || t >= c.kt1 || t < t0 || t >= t1 || t * 16 >= c.size_k) __builtin_trap();
#endif
#ifdef EXL3_ROCM_EXPERIMENT_FAKE_LOAD
            // experiment: no global weight loads (VALU ceiling); words derived from t and the lane
            const bool fake = true;
#else
            const bool fake = false;
#endif
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * SUB_U32);
            if (fake)
            {
                #pragma unroll
                for (int k = 0; k <= NW; ++k) a[k] = (uint32_t) (t * 2654435761u) ^ (lane * 40503u + k * 0x9E3779B9u);
            }
            else if constexpr (half_k)
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
#if EXL3_ROCM_VEC_LOAD
                // integer K: the lane's NW words are contiguous and NW*4-byte aligned -> one vector
                // load (b64 / b96 / b128); the preceding word comes from the neighbouring lane
                // (lane 0: lane 7's last word = the subtile's last word) via a wave shuffle
                if constexpr (NW == 3)
                {
                    typedef __attribute__((address_space(1))) const uint3 gl_u3;
                    gl_u3* vp = (gl_u3*) (p32 + bw);
                    uint3 v; v.x = vp->x; v.y = vp->y; v.z = vp->z;
                    a[1] = v.x; a[2] = v.y; a[3] = v.z;
                    a[0] = __shfl(v.z, (lane + 7) & 7, 8);
                }
                else if constexpr (NW == 4)
                {
                    typedef __attribute__((address_space(1))) const uint4 gl_u4;
                    gl_u4* vp = (gl_u4*) (p32 + bw);
                    uint4 v; v.x = vp->x; v.y = vp->y; v.z = vp->z; v.w = vp->w;
                    a[1] = v.x; a[2] = v.y; a[3] = v.z; a[4] = v.w;
                    a[0] = __shfl(v.w, (lane + 7) & 7, 8);
                }
                else if constexpr (NW == 2)
                {
                    typedef __attribute__((address_space(1))) const uint2 gl_u2;
                    gl_u2* vp = (gl_u2*) (p32 + bw);
                    uint2 v; v.x = vp->x; v.y = vp->y;
                    a[1] = v.x; a[2] = v.y;
                    a[0] = __shfl(v.y, (lane + 7) & 7, 8);
                }
                else
#endif
                {
                    a[0] = p32[bw_prev];
                    #pragma unroll
                    for (int k = 0; k < NW; ++k) a[k + 1] = p32[bw + k];
                }
            }
            if (c.xl)
            {
                // LDS: read through the local address space so the loads lower to ds_load_b128
                // (a pointer that may be either LDS or global is lowered to flat_load)
                typedef __attribute__((address_space(3))) const uint32_t lds_u32;
                lds_u32* xp = (lds_u32*) (const uint32_t*) (x + (t - x_t0) * 16);
                #pragma unroll
                for (int k = 0; k < 8; ++k) xw[k] = xp[k];
                return;
            }
            typedef __attribute__((address_space(1))) const uint4 gl_uint4;
            gl_uint4* xp = (gl_uint4*) (const uint4*) (x + t * 16);
            uint4 x0, x1;
            x0.x = xp[0].x; x0.y = xp[0].y; x0.z = xp[0].z; x0.w = xp[0].w;
            x1.x = xp[1].x; x1.y = xp[1].y; x1.z = xp[1].z; x1.w = xp[1].w;
            xw[0] = x0.x; xw[1] = x0.y; xw[2] = x0.z; xw[3] = x0.w;
            xw[4] = x1.x; xw[5] = x1.y; xw[6] = x1.z; xw[7] = x1.w;
        };

        auto decode_tile = [&](const uint32_t* a, const uint32_t* xw)
        {
#ifdef EXL3_ROCM_EXPERIMENT_NO_DECODE
            // experiment: keep the loads, replace the decode by a trivial consume (bandwidth ceiling)
            { float s = 0.f;
              #pragma unroll
              for (int k = 0; k <= NW; ++k) s += __uint_as_float(a[k] & 0x3FFFFFFFu);
              acc0 += s * __uint_as_float(xw[0] | 0x3F800000u); acc1 += s; return; }
#endif

            if constexpr (cb == 2)
            {
#if !EXL3_ROCM_PAIRED_DOT2
                half xh[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    xh[2 * i]     = __ushort_as_half((unsigned short) (xw[i] & 0xFFFFu));
                    xh[2 * i + 1] = __ushort_as_half((unsigned short) (xw[i] >> 16));
                }
#endif
                // Explicitly interleaved decode: the per-code chain (alignbit -> mad -> mad -> sad
                // -> fma) is 5 dependent ops with ~4-cycle result latency each, and the compiler's
                // schedule left those latencies exposed (~50% VALU utilisation). Processing 8 codes
                // in lock-step, one stage at a time, keeps 8 independent ops between dependent hops
                // (measured 1.5 -> 2.6 T weights/s in micro/decode_rate9.hip). The inline-asm stages
                // are volatile so the compiler cannot re-serialise them per code
                float b0 = 0.f, b1 = 0.f;
                #pragma unroll
                for (int i0 = 0; i0 < 32; i0 += EXL3_ROCM_ILV)
                {
                    uint32_t code[EXL3_ROCM_ILV], t[EXL3_ROCM_ILV];
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; ++j)
                    {
                        const int E = lane_e_rel<bits, half_k>(i0 + j);
                        const int p = (E - 1) >> 5;
                        const int sh = 32 * (p + 1) - E;
                        code[j] = __funnelshift_r(a[p + 1], a[p], sh);
                    }
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; ++j) asm volatile("v_mad_u32_u16 %0, %1, %2, 0" : "=v"(t[j]) : "v"(code[j]), "s"(0xD12Du));
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; ++j) asm volatile("v_mad_u16 %0, %1, %2, %0 op_sel:[0,0,1,1]" : "+v"(t[j]) : "v"(code[j]), "s"(0x83DCu));
#if EXL3_ROCM_PAIRED_DOT2
                    // Paired epilogue: within a group of 8 codes, codes k and k+1 (k even) hit the
                    // adjacent activation rows 2g + 8*((k>>1)&1) + {0,1} = one packed x word, and
                    // they both go to the same output column (k < 4: col 0, else col 1). sad_u8 with
                    // 0x64006400 leaves fp16(1024+bs) in the low half and 0x6400 in the high half;
                    // sad_hi_u8 of the odd code then adds its byte sum into that high half -> one
                    // register holding the fp16 pair, consumed by a single v_dot2_f32_f16.
                    // 4.5 VALU ops per weight instead of 5
                    uint32_t pk[EXL3_ROCM_ILV / 2];
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; j += 2) asm volatile("v_sad_u8 %0, %0, 0, 0x64006400" : "+v"(t[j]));
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; j += 2) asm volatile("v_sad_hi_u8 %0, %1, 0, %2" : "=v"(pk[j / 2]) : "v"(t[j + 1]), "v"(t[j]));
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; j += 2)
                    {
                        const int i = i0 + j;
                        const int g = i >> 3, k = i & 7;
                        const int r = 2 * g + 8 * ((k >> 1) & 1);          // even row of the pair
                        float* acc = (k < 4) ? ((k & 2) ? &b0 : &acc0) : ((k & 2) ? &b1 : &acc1);
                        asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(*acc) : "v"(pk[j / 2]), "v"(xw[r / 2]));
                    }
#else
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; ++j) asm volatile("v_sad_u8 %0, %0, 0, 0x6400" : "+v"(t[j]));
                    #pragma unroll
                    for (int j = 0; j < EXL3_ROCM_ILV; ++j)
                    {
                        const int i = i0 + j;
                        const int g = i >> 3, k = i & 7;
                        const int r = 2 * g + (k & 1) + 8 * ((k >> 1) & 1);
                        const float pr = __half2float(__ushort_as_half((unsigned short) t[j])) * __half2float(xh[r]);
                        if (k < 4) { if (k & 1) b0 += pr; else acc0 += pr; }
                        else       { if (k & 1) b1 += pr; else acc1 += pr; }
                    }
#endif
                }
                acc0 += b0; acc1 += b1;
#if EXL3_ROCM_PAIRED_DOT2
                // sum of the tile's activations (affine correction) straight from the packed pairs
                #pragma unroll
                for (int i = 0; i < 8; ++i) asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(xsum) : "v"(xw[i]), "s"(0x3c003c00u));
#else
                float sx = 0.f;
                #pragma unroll
                for (int i = 0; i < 16; ++i) sx += __half2float(xh[i]);
                xsum += sx;
#endif
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
#if EXL3_ROCM_PIPELINE == 2
        // depth 2: three buffers, loads issued two tiles ahead. Fixed trip count (groups of 3)
        // + explicit tail, and NO divergent branches inside the loop: the prefetch index is
        // clamped to the last tile (a redundant in-bounds load) instead of being predicated.
        // Both the `break`-terminated and the `if (i + k < ntiles)` forms compiled into
        // non-terminating loops (see notes.md)
        uint32_t bufC[NW + 1], xC[8];
        if (ntiles > 0) load_tile(t0, bufA, xA);
        if (ntiles > 1) load_tile(t0 + 1, bufB, xB);
        const int nfull = ntiles / 3;
        const int tlast = t1 - 1;
        for (int g = 0; g < nfull; ++g)
        {
            const int i = g * 3;
            load_tile(t0 + i + 2, bufC, xC);                      // i + 2 < ntiles always holds here
            decode_tile(bufA, xA);
            load_tile(min(t0 + i + 3, tlast), bufA, xA);
            decode_tile(bufB, xB);
            load_tile(min(t0 + i + 4, tlast), bufB, xB);
            decode_tile(bufC, xC);
        }
        // tail (0..2 tiles): already resident in bufA / bufB
        const int ntail = ntiles - nfull * 3;
        if (ntail >= 1) decode_tile(bufA, xA);
        if (ntail >= 2) decode_tile(bufB, xB);
#elif EXL3_ROCM_PIPELINE == 4
        // variant 1 (dual chain): the group's k range is split in two halves that advance in
        // lock-step, each with its own depth-1 prefetch and its own accumulators -> two independent
        // load chains per lane (2 tiles in flight, like depth 2) AND two independent FMA chains.
        // x is fetched just-in-time (LDS / L2), weights are buffered: 4 x (NW+1) VGPRs
        {
            const int half = ntiles >> 1;             // chain 0: [t0, t0+half), chain 1: [t0+half, t1)
            const int n1 = ntiles - half;              // n1 >= half
            uint32_t bufA2[NW + 1], bufB2[NW + 1];
            float acc0b = 0.f, acc1b = 0.f;
            // decode into a second accumulator set: swap, decode, swap back (register renames only)
            auto decode_tile_b = [&](const uint32_t* a, const uint32_t* xw)
            {
                float s0 = acc0, s1 = acc1; acc0 = acc0b; acc1 = acc1b;
                decode_tile(a, xw);
                acc0b = acc0; acc1b = acc1; acc0 = s0; acc1 = s1;
            };
            const int ta0 = t0, tb0 = t0 + half;
            if (half > 0) load_w(ta0, bufA);
            if (n1 > 0) load_w(tb0, bufA2);
            // paired steps over the common length `half` (fixed trip count, no divergent branches)
            const int tlast_a = t0 + half - 1, tlast_b = t1 - 1;
            for (int i = 0; i < half; i += 2)
            {
                load_w(min(ta0 + i + 1, tlast_a), bufB);
                load_w(min(tb0 + i + 1, tlast_b), bufB2);
                load_x(ta0 + i, xA); decode_tile(bufA, xA);
                load_x(tb0 + i, xA); decode_tile_b(bufA2, xA);
                load_w(min(ta0 + i + 2, tlast_a), bufA);
                load_w(min(tb0 + i + 2, tlast_b), bufA2);
                if (i + 1 < half)
                {
                    load_x(ta0 + i + 1, xA); decode_tile(bufB, xA);
                    load_x(tb0 + i + 1, xA); decode_tile_b(bufB2, xA);
                }
            }
            // chain 1 is at most one tile longer
            if (n1 > half) { load_x(tb0 + half, xA); decode_tile_b(bufA2, xA); }
            acc0 += acc0b; acc1 += acc1b;
        }
#elif EXL3_ROCM_PIPELINE == 6
        // variant 6: weights two tiles ahead with a 2-buffer ring (depth 2 with only 2 buffers:
        // buffer A is reloaded right after its decode, so the load for tile t+2 is in flight during
        // the decode of t+1) and x just-in-time; two accumulator pairs alternate per tile
        {
            float acc0b = 0.f, acc1b = 0.f;
            auto decode_tile_b = [&](const uint32_t* a, const uint32_t* xw)
            {
                float s0 = acc0, s1 = acc1; acc0 = acc0b; acc1 = acc1b;
                decode_tile(a, xw);
                acc0b = acc0; acc1b = acc1; acc0 = s0; acc1 = s1;
            };
            const int tlast = t1 - 1;
            if (ntiles > 0) load_w(t0, bufA);
            if (ntiles > 1) load_w(t0 + 1, bufB);
            const int npairs = ntiles / 2;
            for (int g = 0; g < npairs; ++g)
            {
                const int i = g * 2;
                load_x(t0 + i, xA); decode_tile(bufA, xA);
                load_w(min(t0 + i + 2, tlast), bufA);
                load_x(t0 + i + 1, xA); decode_tile_b(bufB, xA);
                load_w(min(t0 + i + 3, tlast), bufB);
            }
            if (ntiles & 1) { load_x(t1 - 1, xA); decode_tile(bufA, xA); }
            acc0 += acc0b; acc1 += acc1b;
        }
#elif EXL3_ROCM_PIPELINE == 5
        // variant 5: exactly the default depth-1 loop, but x just-in-time (diagnostic)
        if (ntiles > 0) load_w(t0, bufA);
        for (int i = 0; i < ntiles; i += 2)
        {
            const bool has_b = (i + 1 < ntiles);
            if (has_b) load_w(t0 + i + 1, bufB);
            load_x(t0 + i, xA);
            decode_tile(bufA, xA);
            if (i + 2 < ntiles) load_w(t0 + i + 2, bufA);
            if (has_b) { load_x(t0 + i + 1, xA); decode_tile(bufB, xA); }
        }
#elif EXL3_ROCM_PIPELINE == 9
        // variant 9: default loop, load_tile for the weights (x part discarded) + x just-in-time via
        // load_tile as well (diagnostic: lambda identity vs. schedule)
        if (ntiles > 0) load_tile(t0, bufA, xA);
        for (int i = 0; i < ntiles; i += 2)
        {
            const bool has_b = (i + 1 < ntiles);
            if (has_b) load_tile(t0 + i + 1, bufB, xB);
            decode_tile(bufA, xA);
            if (i + 2 < ntiles) load_tile(t0 + i + 2, bufA, xA);
            if (has_b) decode_tile(bufB, xB);
        }
#elif EXL3_ROCM_PIPELINE == 8
        // variant 8: default depth-1 loop with load_tile (w + x buffered) but the `min` clamp form
        // instead of predicated loads (diagnostic: is it the predication or the x path?)
        {
            const int tlast = t1 - 1;
            if (ntiles > 0) load_tile(t0, bufA, xA);
            const int npairs = ntiles / 2;
            for (int g = 0; g < npairs; ++g)
            {
                const int i = g * 2;
                load_tile(t0 + i + 1, bufB, xB);
                decode_tile(bufA, xA);
                load_tile(min(t0 + i + 2, tlast), bufA, xA);
                decode_tile(bufB, xB);
            }
            if (ntiles & 1) decode_tile(bufA, xA);
        }
#elif EXL3_ROCM_PIPELINE == 7
        // variant 7: as 6 but a single accumulator pair
        {
            const int tlast = t1 - 1;
            if (ntiles > 0) load_w(t0, bufA);
            if (ntiles > 1) load_w(t0 + 1, bufB);
            const int npairs = ntiles / 2;
            for (int g = 0; g < npairs; ++g)
            {
                const int i = g * 2;
                load_x(t0 + i, xA); decode_tile(bufA, xA);
                load_w(min(t0 + i + 2, tlast), bufA);
                load_x(t0 + i + 1, xA); decode_tile(bufB, xA);
                load_w(min(t0 + i + 3, tlast), bufB);
            }
            if (ntiles & 1) { load_x(t1 - 1, xA); decode_tile(bufA, xA); }
        }
#elif EXL3_ROCM_PIPELINE == 3
        // variant 3: weights two tiles ahead (3 weight buffers), x fetched just-in-time (LDS slice
        // or L2-resident A: short latency, so no need to buffer it) -> saves 2 x 8 VGPRs vs depth 2
        uint32_t bufC[NW + 1];
        if (ntiles > 0) load_w(t0, bufA);
        if (ntiles > 1) load_w(t0 + 1, bufB);
        const int nfull = ntiles / 3;
        const int tlast = t1 - 1;
        for (int g = 0; g < nfull; ++g)
        {
            const int i = g * 3;
            load_w(t0 + i + 2, bufC);
            load_x(t0 + i, xA);
            decode_tile(bufA, xA);
            load_w(min(t0 + i + 3, tlast), bufA);
            load_x(t0 + i + 1, xA);
            decode_tile(bufB, xA);
            load_w(min(t0 + i + 4, tlast), bufB);
            load_x(t0 + i + 2, xA);
            decode_tile(bufC, xA);
        }
        const int ntail = ntiles - nfull * 3;
        if (ntail >= 1) { load_x(t0 + nfull * 3, xA); decode_tile(bufA, xA); }
        if (ntail >= 2) { load_x(t0 + nfull * 3 + 1, xA); decode_tile(bufB, xA); }
#else
        if (ntiles > 0) load_tile(t0, bufA, xA);
        for (int i = 0; i < ntiles; i += 2)
        {
            const bool has_b = (i + 1 < ntiles);
            if (has_b) load_tile(t0 + i + 1, bufB, xB);
#if EXL3_ROCM_SCHED_BARRIER
            // variant 4: keep the prefetch (VMEM) ahead of the decode (VALU) - the compiler is
            // otherwise free to sink the loads towards their uses
            __builtin_amdgcn_sched_barrier(0);
#endif
            decode_tile(bufA, xA);
            if (i + 2 < ntiles) load_tile(t0 + i + 2, bufA, xA);
#if EXL3_ROCM_SCHED_BARRIER
            __builtin_amdgcn_sched_barrier(0);
#endif
            if (has_b) decode_tile(bufB, xB);
        }
#endif
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
#if EXL3_ROCM_DET_KSPLIT
            float* slot = kslots + (size_t) idx * 16;   // idx = ks * items + item
            slot[lane] = acc0;
            slot[lane + 8] = acc1;
#else
            atomicAdd(out + lane, acc0);
            atomicAdd(out + lane + 8, acc1);
#endif
        }
    }
#if EXL3_ROCM_DET_KSPLIT
    if (KS > 1)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < items * 16; i += blockDim.x)
        {
            const int item = i >> 4, e = i & 15;
            float s = 0.f;
            for (int ks = 0; ks < KS; ++ks) s += kslots[(size_t) (ks * items + item) * 16 + e];
            const int s_ = item % c.subs_tile, m = item / c.subs_tile;
            c.sh_c[(size_t) m * cols + (size_t) s_ * 16 + e] = s;
        }
    }
#endif
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

// Ordered-partials epilogue capacity test, shared by the inner (device) and the host grid bound:
// both must agree, otherwise the host could offer a grid beyond co-residency to a lock-path call
__host__ __device__ __forceinline__ int rocm_ordered_per_col(int num_slices, int tiles_n)
{
    return (num_slices + tiles_n - 1) / tiles_n + 1;
}
__host__ __device__ __forceinline__ bool rocm_ordered_fits(int num_slices, int tiles_n, int size_m, int ts_n)
{
    if (size_m > 16) return false;
    const int per_col = rocm_ordered_per_col(num_slices, tiles_n);
    if (per_col > ROCM_PARTIALS_PER_COL) return false;
    return (size_t) tiles_n * per_col * size_m * ts_n <= (size_t) ROCM_PARTIALS_FLOATS;
}

// shared inner entry point; C is [m][n] with row stride size_n. post_scale
// applies only when shmem_out_had is set

// lane_only: decode instance (size_m <= LANE_TIER_MAX_M guaranteed by the host); leaving the warp/dq
// tier out of the kernel saves ~14 VGPRs (113 -> 99 at 3bpw), i.e. 3 -> 4 blocks/CU at 512 threads
template<EXL3_GEMM_T_ARGS, bool shmem_out_had, bool dual = false, bool lane_only = false>
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
    float* __restrict__ sh = nullptr,
    const half* pre_scale = nullptr,  // fused input Hadamard: A is the RAW input, rotated per segment with this scale
    int* __restrict__ lock_base = nullptr,  // ordered epilogue: base of the device lock buffer (partials + tickets); nullptr = column locks
    // dual GEMM (gate/up fusion): a second weight matrix of the SAME k, n and K sharing A/suh,
    // presented as column tiles [tiles_n, 2*tiles_n) of one virtual matrix. B1/C1/post_scale1
    // replace B/C/post_scale for those tiles; C1 has the same row stride as C
    const uint16_t* __restrict__ B1 = nullptr,
    void* __restrict__ C1 = nullptr,
    const half* post_scale1 = nullptr,
    const half* pre_scale1 = nullptr,  // the second matrix's own input scale (gate/up do not share suh)
    int size_n1 = 0                    // second matrix width (0: = size_n). Same k and K
)
{
    using namespace exl3_rocm_inner;

    if (size_n_stride == 0) size_n_stride = size_n;
    const int n_full = size_n_stride;    // B rows and C rows span the full matrix width
    constexpr int TS_N = TILESIZE_N;
    float* sh_c = sh;

    int tiles_k = size_k / 16;                 // trellis k-subtiles (16 wide)
    const int tiles_n1 = size_n / TS_N;        // column tiles of the first matrix
    if (size_n1 == 0) size_n1 = size_n;
    int tiles_n = dual ? tiles_n1 + size_n1 / TS_N : tiles_n1;
    int units = tiles_k * tiles_n;
    const int nsub0 = n_full / 16, nsub1 = size_n1 / 16;   // B row widths (subtiles) per matrix
    const uint16_t* B0 = B; void* C0 = C; const half* post_scale0 = post_scale; const half* pre_scale0 = pre_scale;
    int num_slices = gridDim.x;
#if EXL3_ROCM_SLICE_PERMUTE
    // Blocks are dispatched in index order and adjacent indices take adjacent (column, k) slices
    // -> at any moment the resident blocks stream neighbouring rows of B and their DRAM traffic
    // piles onto the same channels (measured: identical work per block, duration growing
    // linearly with blockIdx). Stride the slice assignment so co-resident blocks are spread over
    // the whole tile space
    const int stride = 48;   // ~ CU count; coprime with grid sizes that are not multiples of 48 handled below
    int sid = blockIdx.x;
    if (num_slices % stride == 0) sid = (blockIdx.x % stride) * (num_slices / stride) + blockIdx.x / stride;
#else
    const int sid = blockIdx.x;
#endif
    int beg = (int) ((int64_t) units * sid / num_slices);
    int end = (int) ((int64_t) units * (sid + 1) / num_slices);

#if EXL3_ROCM_ORDERED_EPILOGUE
    // Ordered-partials epilogue: every slice stores its fp32 partial tile to partials[sid] and
    // takes a ticket on the column; the LAST arriver sums the column's partials in slice order
    // (deterministic) and writes the output. No block ever waits for another one -> no
    // co-residency requirement, and the k-chain of dependent round trips becomes one. One
    // 16-row slab only (the partials are indexed by slice, not by slab): the host passes
    // lock_base only when size_m <= 16
    // slot layout: [col][ordinal within the column][size_m x TS_N]; at most ROCM_PARTIALS_PER_COL
    // slices per column (the host caps the grid so that ceil(num_slices / tiles_n) + 1 fits).
    // Compact and reused across calls -> stays L2-resident (a per-slice layout over the whole
    // buffer was ~2 us slower per GEMM: cold lines)
    const int per_col = rocm_ordered_per_col(num_slices, tiles_n);
    const bool ordered = (lock_base != nullptr) && rocm_ordered_fits(num_slices, tiles_n, size_m, TS_N);
    float* partials = ordered ? (float*) (lock_base + ROCM_PARTIALS_OFFSET) : nullptr;
    int* tickets = ordered ? (lock_base + ROCM_TICKETS_OFFSET) : nullptr;
    const int part_stride = size_m * TS_N;   // floats per slot
#else
    const bool ordered = false;
#endif

    SegCtx c;
    c.A = A;
    c.B = (const uint32_t*) B;
    c.sh_c = sh_c;
    c.size_m = size_m;
    c.size_k = size_k;
    c.nsub_total = n_full / 16;
    c.subs_tile = TS_N / 16;

#ifdef EXL3_ROCM_PROBE
    // debug: per-block phase timestamps -> locks + 65536 + blockIdx.x * 8 (uint64 x 4)
    unsigned long long* probe = (unsigned long long*) (locks + 65536) + blockIdx.x * 4;
    unsigned long long tp0 = __probe_now();
    unsigned long long tp_start = tp0;
    unsigned long long tp_phase1 = 0, tp_lock = 0, tp_out = 0;
#endif
    while (beg < end)
    {
        int col = beg / tiles_k;
        int seg_k0 = beg % tiles_k;
        int seg_k1 = ((end - 1) / tiles_k == col) ? ((end - 1) % tiles_k) + 1 : tiles_k;

        c.kt0 = seg_k0;
        c.kt1 = seg_k1;
        // dual GEMM: virtual column tile -> (matrix, column tile within it). Compile-time: the
        // extra live pointers pushed the single kernel into scratch spills (and scratch lowers the
        // real co-residency below the probe's measurement -> lock-path deadlock)
        const int vcol = col;
        if constexpr (dual)
        {
            const bool second = vcol >= tiles_n1;
            col = second ? vcol - tiles_n1 : vcol;
            c.B = (const uint32_t*) (second ? B1 : B0);
            C = second ? C1 : C0;
            post_scale = second ? post_scale1 : post_scale0;
            pre_scale = second ? pre_scale1 : pre_scale0;
            c.nsub_total = second ? nsub1 : nsub0;
        }
        // row stride of B / C / post_scale for the matrix of this column tile
        const int n_full_c = (dual && vcol >= tiles_n1) ? size_n1 : n_full;
        c.col = col;
        int cols = c.subs_tile * 16;

#if EXL3_ROCM_FUSED_HAD
        // Rotate this segment's x slice (size_m rows x (seg_k1 - seg_k0) tiles) into LDS: the
        // 128-wide Hadamard blocks are 8 tiles, and segment bounds are tile-aligned, so the slice
        // is widened to 128-aligned bounds. pre_scale = suh. One warp per 128-block
        if (pre_scale)
        {
            const int kb0 = (seg_k0 * 16) / 128, kb1 = (seg_k1 * 16 + 127) / 128;
            const int nblk = kb1 - kb0;
            half* xl = (half*) (sh_c + 16 * TS_N);
            const int seg_k = (seg_k1 - seg_k0) * 16;
            // host-side grid bound guarantees this; a violation would silently corrupt sh_c
            if (size_m * seg_k > EXL3_X_LDS_TILES * 16) __builtin_trap();
            const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
            __syncthreads();   // previous segment's readers of xl / sh_c are done
            // whole-wave iteration count so the shuffle-based Hadamard never runs with a partial wave
            for (int slot = warp; slot < size_m * nblk; slot += nwarps)
            {
                const int m = slot / nblk, b = kb0 + slot % nblk;
                const half4 v = ((const half4*) (A + (size_t) m * size_k + b * 128))[lane];
                const half4 sc = ((const half4*) (pre_scale + b * 128))[lane];
                half2 vx = __hmul2(v.x, sc.x), vy = __hmul2(v.y, sc.y);
                float v0 = __half2float(__low2half(vx)), v1 = __half2float(__high2half(vx));
                float v2 = __half2float(__low2half(vy)), v3 = __half2float(__high2half(vy));
                float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
                float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
                shuffle_had_f4x32(h0, h1, h2, h3, lane);
                const float rs = 0.088388347648f;
                // element index within the segment slice; the 128-block may overhang the segment
                const int kk = b * 128 + lane * 4 - seg_k0 * 16;
                if (kk >= 0 && kk < seg_k)
                {
                    half* dst = xl + (size_t) m * seg_k + kk;
                    dst[0] = __float2half(h0 * rs); dst[1] = __float2half(h1 * rs);
                    dst[2] = __float2half(h2 * rs); dst[3] = __float2half(h3 * rs);
                }
            }
            __syncthreads();
            c.xl = xl;
        }
        else c.xl = nullptr;
#else
        c.xl = nullptr;
#endif

        // Lane-local tiers re-decode the weights once per m row, the warp tier once per
        // subtile: the former wins for decode-shaped m, the latter for prefill chunks
        constexpr int LANE_TIER_MAX_M = 8;
        // (the host only enables the fused Hadamard for size_m <= LANE_TIER_MAX_M, whose tier
        // reads c.xl; the other tiers read the pre-rotated A)
        if constexpr (cb == 2 && bits == 4 && !half_k && !EXL3_ROCM_LANE_TIER_ALL && !lane_only)
        {
            phase1_b4c2(c);
        }
        else if constexpr (cb == 2 && bits == 6 && !half_k && !EXL3_ROCM_LANE_TIER_ALL && !lane_only)
        {
            phase1_b6c2(c);
        }
        else if constexpr (bits > 0)
        {
            if constexpr (lane_only)
                phase1_lane<bits, half_k, cb>(c);
            else if (c.size_m <= LANE_TIER_MAX_M)
                phase1_lane<bits, half_k, cb>(c);
            else
                phase1_dq<bits, cb, half_k>(c);
        }
        // bits == 0 never reaches the inner (the caller switches on K first)

        __syncthreads();
#ifdef EXL3_ROCM_PROBE
        unsigned long long tpa = __probe_now();
#endif

        int lock_i = tiles_k - seg_k1;
        int lock_d = seg_k1 - seg_k0;
        int* lock = &locks[vcol];
        bool first = (lock_i == 0);
        bool last = (lock_i + lock_d == tiles_k);

#if EXL3_ROCM_ORDERED_EPILOGUE
        if (ordered)
        {
            const bool whole_column = first && last;   // single slice covers the column: no partials
            if (!whole_column)
            {
                // slices covering this column: sid_lo..sid_hi (contiguous in slice id)
                const int col_beg = vcol * tiles_k, col_end = col_beg + tiles_k;
                int sid_lo = (int) (((int64_t) col_beg * num_slices) / units);          // beg(sid_lo) <= col_beg
                while ((int) ((int64_t) units * (sid_lo + 1) / num_slices) <= col_beg) ++sid_lo;
                int sid_hi = (int) (((int64_t) (col_end - 1) * num_slices) / units);    // beg(sid_hi) <= col_end - 1
                while ((int) ((int64_t) units * (sid_hi + 1) / num_slices) < col_end) ++sid_hi;
                const int nsl = sid_hi - sid_lo + 1;
                // publish this slice's partial to slot (col, ordinal), take the column ticket
                float* col_slots = partials + (size_t) vcol * per_col * part_stride;
                float* mine = col_slots + (size_t) (sid - sid_lo) * part_stride;
                for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x) mine[i] = sh_c[i];
                // agent-scope release fence (the ticket RMW below is acq_rel at agent scope, but
                // the partial stores by the OTHER threads must be ordered before it: block barrier
                // then a single release fence by the ticket thread)
                __syncthreads();
                // ticket broadcast through the k-split slot area (free after phase 1; keeps the
                // static LDS size unchanged)
                int& s_ticket = *(int*) (sh_c + 16 * TS_N + (EXL3_ROCM_FUSED_HAD ? EXL3_X_LDS_FLOATS : 0));
                if (threadIdx.x == 0)
                {
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
                    s_ticket = __hip_atomic_fetch_add(tickets + vcol, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
                __syncthreads();
                if (s_ticket != nsl - 1)
                {
#ifdef EXL3_ROCM_PROBE
                    { unsigned long long tpq = __probe_now(); tp_phase1 += tpa - tp0; tp_lock += tpq - tpa; tp0 = tpq; }
#endif
                    beg = (vcol + 1) * tiles_k;   // not the last arriver: done with this column
                    continue;
                }
                // last arriver: acquire (only this block pays the L0/L1 invalidate), reset the
                // ticket, sum the partials in slice order
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
                if (threadIdx.x == 0) __hip_atomic_store(tickets + vcol, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                // the acquire fence above made the other slices' stores visible; plain loads, all
                // nsl rows independent -> issued together
                for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
                {
                    float s = col_slots[i];
                    #pragma unroll 4
                    for (int q = 1; q < nsl; ++q) s += col_slots[(size_t) q * part_stride + i];
                    sh_c[i] = s;
                }
                __syncthreads();
            }
            first = true; last = true;   // fall through to the final-block output path below
        }
        else
#endif
        lock_acquire(lock, lock_i);
#ifdef EXL3_ROCM_PROBE
        unsigned long long tpb = __probe_now();   // ordered: includes partial publish + ticket + (last) gather
#endif

        if (!first)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    sh_c[i] += ((const float*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n];
                else
                    sh_c[i] += __half2float(((const half*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n]);
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
                    ((float*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
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
                    const half* sb = post_scale + ((size_t) col * TS_N + b * 128) % n_full_c;
                    h0 *= rs * __half2float(sb[lane * 4 + 0]);
                    h1 *= rs * __half2float(sb[lane * 4 + 1]);
                    h2 *= rs * __half2float(sb[lane * 4 + 2]);
                    h3 *= rs * __half2float(sb[lane * 4 + 3]);
                }
                else
                {
                    h0 *= rs; h1 *= rs; h2 *= rs; h3 *= rs;
                }

                size_t n0 = (size_t) m * n_full_c + (size_t) col * TS_N + b * 128 + lane * 4;
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
                    ((float*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full_c + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }

#if EXL3_ROCM_ORDERED_EPILOGUE
        if (!ordered)
#endif
        lock_release(lock, lock_d, last);
#ifdef EXL3_ROCM_PROBE
        unsigned long long tpc = __probe_now();
        tp_phase1 += tpa - tp0; tp_lock += tpb - tpa; tp_out += tpc - tpb; tp0 = tpc;
#endif

        // phase 1 must store fresh values: the accumulate path adds into sh_c
        beg = (vcol + 1) * tiles_k;
    }
#ifdef EXL3_ROCM_PROBE
    if (threadIdx.x == 0) { probe[0] = tp_phase1; probe[1] = tp_lock; probe[2] = tp_start; probe[3] = __probe_now(); }
#endif
}
