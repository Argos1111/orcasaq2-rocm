#pragma once

// HIP compat for the shared headers. The top shims are visible to every compilation
// pass (hipcc's device pass also checks __host__ bodies); the device polyfills below
// are HIP-compiler-only, since a standard host compiler must not see them.

#include <hip/hip_runtime.h>

// host parses need these typedefs; HIP's headers define them for HIP passes only
#ifndef __align__
#define __align__(x) __attribute__((aligned(x)))
#endif
#if !defined(__HIPCC__)
// declaration for host parses; the device pass gets the header's own
__half2 __halves2half2(__half a, __half b);
#endif

// clang's HIP math declares rsqrtf __device__-only, so host callers need this overload
__host__ inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }

#if !defined(USE_ROCM) || defined(__HIPCC__)

#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#define __shfl_sync(mask, var, ...) __shfl(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...) __shfl_down(var, __VA_ARGS__)
#define __shfl_up_sync(mask, var, ...) __shfl_up(var, __VA_ARGS__)
#define __ballot_sync(mask, ...) __ballot(__VA_ARGS__)

namespace polyfill
{

__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c)
{
    // every call site in this codebase passes unsigned byte lanes (codebook decode), so the
    // u8 dot is exact; gfx9+/gfx10+/gfx11+ have it as v_dot4_u32_u8
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
    return (int) __builtin_amdgcn_udot4(a, b, (uint32_t) c, false);
#else
    int result = c;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t va = static_cast<uint32_t>(static_cast<uint8_t>((a >> (i * 8)) & 0xFF));
        uint32_t vb = static_cast<uint32_t>(static_cast<uint8_t>((b >> (i * 8)) & 0xFF));
        result += va * vb;
    }
    return result;
#endif
}

__device__ __forceinline__ uint32_t dp4a(uint32_t a, uint32_t b, uint32_t c)
{
    return static_cast<uint32_t>(dp4a(a, b, static_cast<int>(c)));
}

__device__ __forceinline__ half2 hmax2(half2 a, half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ half2 hmin2(half2 a, half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

#if defined(__HIPCC__)
__device__ __forceinline__ __hip_bfloat16 float2bfloat16_rz(float f)
{
    uint32_t u = __float_as_uint(f);
    uint16_t r = static_cast<uint16_t>(u >> 16);
    return __ushort_as_bfloat16(r);
}

__device__ __forceinline__ __hip_bfloat16 float2bfloat16_rn(float f)
{
    return __float2bfloat16(f);
}
#endif

} // namespace polyfill

#ifndef __dp4a
#define __dp4a polyfill::dp4a
#endif
#ifndef __hmax2
#define __hmax2 polyfill::hmax2
#endif
#ifndef __hmin2
#define __hmin2 polyfill::hmin2
#endif
#if defined(__HIPCC__)
#ifndef __float2bfloat16_rz
#define __float2bfloat16_rz polyfill::float2bfloat16_rz
#endif
#ifndef __float2bfloat16_rn
#define __float2bfloat16_rn polyfill::float2bfloat16_rn
#endif
#endif

// HIP's fp16 header only provides __ldcg for __half/__half2; CUDA's is generic over the
// scalar types. Cache-hint-free loads are the correct semantics on AMD (no L1 bypass to
// express), so plain dereferences
#if defined(__HIPCC__)
__device__ __forceinline__ float __ldcg(const float* p) { return *p; }
__device__ __forceinline__ int __ldcg(const int* p) { return *p; }
__device__ __forceinline__ unsigned int __ldcg(const unsigned int* p) { return *p; }
__device__ __forceinline__ float2 __ldcg(const float2* p) { return *p; }
__device__ __forceinline__ float4 __ldcg(const float4* p) { return *p; }
__device__ __forceinline__ int4 __ldcg(const int4* p) { return *p; }
__device__ __forceinline__ uint4 __ldcg(const uint4* p) { return *p; }
#endif

// mask dropped (call sites are full-warp); HIP's masked 64-bit __syncwarp
// hardware-faults on wave32 gfx1100
#define __syncwarp(...) __syncwarp()

// gfx1100: 64 KB LDS per block with no opt-in beyond it (quantize_tiles' K = 2
// cost tables gate on this)
#define QUANTIZE_TILES_SMEM_LIMIT 65536

#endif  // !defined(USE_ROCM) || defined(__HIPCC__)
