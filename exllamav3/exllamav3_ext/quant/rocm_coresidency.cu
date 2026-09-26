// ROCm: empirical co-residency bound for plain-launched kernels that synchronise across blocks.
//
// The body GEMM's column lock spins on other blocks of the same grid, so every block must be
// resident at once. cudaLaunchCooperativeKernel would guarantee that but costs ~20 us per launch
// on HIP (130 us in a graph), so the fork uses plain launches bounded by an occupancy estimate.
// cudaOccupancyMaxActiveBlocksPerMultiprocessor is NOT that bound on gfx1100: measured against a
// spin-chain probe it under-reports (3 vs 4 at 512 thr / <100 VGPR, 3 vs 6 at 256 thr) and
// over-reports (3 vs 2 at 512 thr / 125 VGPR) - the latter deadlocked the GPU and froze the host.
//
// Here the bound is measured. The target kernel has side effects, so a stand-in is launched: a
// ladder of probe kernels spanning the VGPR range, of which the first with >= the target's VGPR
// count (cudaFuncGetAttributes) is run at the target's block size and LDS (as dynamic LDS). The
// probe is a chain (block b waits for block b+1) that completes iff all N blocks are resident;
// N is raised one block/CU at a time until it fails. Result minus one block/CU of margin is the
// grid cap. Cached per (device, kernel, block_dim); ~50-200 ms per distinct kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "rocm_coresidency.cuh"
#include <map>
#include <mutex>
#include <tuple>
#include <cstdio>

#if defined(USE_ROCM)

namespace
{

// Chain probe: block b spins until block b+1 has arrived. Completes iff all N blocks are resident.
// NV live VGPRs are pinned through asm barriers; the dynamic LDS reproduces the target's LDS.
template<int NV>
__global__ void coresidency_probe(unsigned* flags, unsigned N, unsigned* ok, float* sink, unsigned spin_limit)
{
    extern __shared__ float sh[];
    float v[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) { v[i] = sink[(threadIdx.x + i) & 1023]; asm volatile("" : "+v"(v[i])); }
    const unsigned b = blockIdx.x;
    bool timed_out = false;
    if (threadIdx.x == 0)
    {
        unsigned spins = 0;
        if (b + 1 < N)
            while (__hip_atomic_load(&flags[b + 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0)
                if (++spins > spin_limit) { timed_out = true; break; }
        __hip_atomic_store(&flags[b], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (!timed_out) atomicAdd(ok, 1u);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NV; ++i) { asm volatile("" : "+v"(v[i])); s += v[i]; }
    sh[threadIdx.x] = s;
    __syncthreads();
    if (sh[(threadIdx.x + 1) % blockDim.x] == 12345.f) sink[0] = s;   // keep the array live
}

struct ProbeEntry { const void* fn; int num_regs; };

// VGPR ladder (numRegs reported by cudaFuncGetAttributes = NV + ~4)
#define PROBE(n) { (const void*) coresidency_probe<n>, 0 }
ProbeEntry g_probes[] =
{
    PROBE(8),   PROBE(16),  PROBE(24),  PROBE(32),  PROBE(40),  PROBE(48),  PROBE(56),  PROBE(64),
    PROBE(72),  PROBE(80),  PROBE(88),  PROBE(96),  PROBE(104), PROBE(112), PROBE(120), PROBE(128),
    PROBE(136), PROBE(144), PROBE(152), PROBE(160), PROBE(168), PROBE(176), PROBE(184), PROBE(192),
    PROBE(208), PROBE(224), PROBE(240), PROBE(248)
};
#undef PROBE
constexpr int NUM_PROBES = sizeof(g_probes) / sizeof(g_probes[0]);

std::mutex g_mtx;
std::map<std::tuple<int, const void*, int>, int> g_cache;   // (device, kernel, block_dim) -> blocks/CU
bool g_probes_init = false;
unsigned* g_scratch[16] = {};

void init_probes()
{
    if (g_probes_init) return;
    for (int i = 0; i < NUM_PROBES; ++i)
    {
        cudaFuncAttributes fa;
        if (cudaFuncGetAttributes(&fa, g_probes[i].fn) == cudaSuccess) g_probes[i].num_regs = fa.numRegs;
        else g_probes[i].num_regs = 1 << 20;
    }
    cudaGetLastError();
    g_probes_init = true;
}

}  // namespace

int rocm_coresident_blocks_per_cu(const void* kernel, int block_dim, int dyn_smem)
{
    int device = 0;
    cudaGetDevice(&device);
    std::lock_guard<std::mutex> lock(g_mtx);
    auto key = std::make_tuple(device, kernel, block_dim);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    init_probes();

    cudaFuncAttributes fa;
    cuda_check(cudaFuncGetAttributes(&fa, kernel));
    const int target_regs = fa.numRegs;
    const int target_lds = (int) fa.sharedSizeBytes + dyn_smem;

    // smallest probe with >= the target's VGPR count (conservative: more registers = fewer waves)
    const void* probe = nullptr;
    for (int i = 0; i < NUM_PROBES; ++i)
        if (g_probes[i].num_regs >= target_regs) { probe = g_probes[i].fn; break; }
    if (!probe) probe = g_probes[NUM_PROBES - 1].fn;

    int num_cus = 0;
    cuda_check(cudaDeviceGetAttribute(&num_cus, cudaDevAttrMultiProcessorCount, device));

    // API estimate as a starting point / fallback
    int api_bps = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api_bps, kernel, block_dim, dyn_smem);
    cudaGetLastError();

    static const bool verbose = getenv("EXL3_ROCM_CORESIDENCY_VERBOSE") != nullptr;
    static const bool disabled = getenv("EXL3_ROCM_CORESIDENCY_OFF") != nullptr;
    if (disabled)
    {
        g_cache[key] = MAX(api_bps, 1);
        return g_cache[key];
    }

    if (!g_scratch[device])
    {
        cuda_check(cudaMalloc(&g_scratch[device], (8192 + 1 + 1024) * sizeof(unsigned)));
    }
    unsigned* flags = g_scratch[device];
    unsigned* ok = flags + 8192;
    float* sink = (float*) (ok + 1);
    cuda_check(cudaMemset(sink, 0, 1024 * sizeof(float)));

    cudaStream_t s;
    cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX(target_lds, 1));
    cudaGetLastError();

    // Increase until the chain fails. spin_limit is short: a probe that fails does so by timing
    // out, so this must terminate quickly without a scheduler dependence on the other blocks.
    // ~2e6 spins ~ tens of ms.
    int real = 0;
    const int max_try = 16;
    for (int b = 1; b <= max_try; ++b)
    {
        unsigned N = (unsigned) (num_cus * b);
        if (N > 8192) break;
        cuda_check(cudaMemsetAsync(flags, 0, 8192 * sizeof(unsigned), s));
        cuda_check(cudaMemsetAsync(ok, 0, sizeof(unsigned), s));
        void* args[] = { &flags, &N, &ok, &sink, nullptr };
        unsigned spin_limit = 2000000u;
        args[4] = &spin_limit;
        cudaError_t e = cudaLaunchKernel(probe, dim3(N), dim3(block_dim), args, MAX(target_lds, 1), s);
        if (e != cudaSuccess) { cudaGetLastError(); break; }
        cuda_check(cudaStreamSynchronize(s));
        unsigned h = 0;
        cuda_check(cudaMemcpy(&h, ok, sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (h == N) real = b; else break;
    }
    cudaStreamDestroy(s);

    // The probe reproduces block size, LDS and (>=) VGPRs, so the measurement is exact or
    // conservative; EXL3_ROCM_CORESIDENCY_MARGIN=n keeps n blocks/CU below it
    static const int margin = getenv("EXL3_ROCM_CORESIDENCY_MARGIN") ? atoi(getenv("EXL3_ROCM_CORESIDENCY_MARGIN")) : 0;
    int bound = MAX(real - margin, 1);
    if (verbose)
        fprintf(stderr, "[exl3 rocm] coresidency kernel=%p block=%d regs=%d lds=%d: api=%d measured=%d -> using %d blocks/CU\n",
            kernel, block_dim, target_regs, target_lds, api_bps, real, bound);
    g_cache[key] = bound;
    return bound;
}

#endif
