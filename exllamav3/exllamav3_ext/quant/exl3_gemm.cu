#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "bits_k.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv.cuh"
#include "exl3_gemv_int8.cuh"
#include "coop_autotune.cuh"
#include "rocm_coresidency.cuh"
#include <set>
#include <vector>

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
    return x + 1;
}

#if defined(USE_ROCM)
__global__ __launch_bounds__(256)
void exl3_gemm_had_kernel
(
    const half* __restrict__ A,
    half* __restrict__ A_had,
    const half* __restrict__ suh,
    const int size_k,
    const int total_warps
)
{
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    if (this_warp >= total_warps) return;
    had_hf_r_128_inner<true, false>
    (
        A + this_warp * 128,
        A_had + this_warp * 128,
        suh + (this_warp * 128) % size_k,
        0.088388347648f  // 1/sqrt(128)
    );
}
#endif

#if defined(USE_ROCM)
// Fused input Hadamard: every block rotates its k segment (widened to 128-aligned bounds) into an
// LDS slice of EXL3_X_LDS_TILES tiles. A segment never spans two column tiles, so its length is
// min(tiles_k, ceil(units / grid)); this returns the smallest grid for which the padded slice
// fits, or -1 when even a full-k segment does not fit (then the separate Hadamard kernel runs)
static inline int exl3_rocm_fused_had_min_grid(int size_m, int size_k, int size_n, int tilesize_n)
{
#if !EXL3_ROCM_FUSED_HAD
    return -1;
#endif
    if (size_m > 8) return -1;                           // only the lane tier reads the LDS slice
    const int tiles_k = size_k / 16;
    const int units = tiles_k * (size_n / tilesize_n);
    const int budget_tiles = EXL3_X_LDS_TILES / size_m;
    const int max_seg = budget_tiles - 16;               // 128-alignment can add up to 7 tiles each side
    if (max_seg < 1) return -1;
    if (tiles_k <= max_seg) return 1;
    return CEIL_DIVIDE(units, max_seg);
}
#endif

// The ROCm body kernel stages everything in static LDS (inner_sh); requesting SMEM_MAX of dynamic
// LDS on top only lowers occupancy
#if defined(USE_ROCM)
#define GEMM_DYN_SMEM 0
#else
#define GEMM_DYN_SMEM SMEM_MAX
#endif

#if defined(USE_ROCM)
// Grid bound (blocks per CU) for the plain-launched body kernel. With the ordered-partials epilogue
// (EXL3_ROCM_ORDERED_EPILOGUE, size_m <= 16) no block waits for another, so the grid may exceed the
// co-residency limit; EXL3_ROCM_OVERSUB (default 1: grids beyond the bound measured slower in the
// real decode chain on gfx1100 even when they win in isolation) x the measured bound is offered to the
// autotuner. Multi-slab calls (size_m > 16) keep the column locks and the hard bound
static inline int rocm_gemm_blocks_per_cu(const void* kernel, int block_dim, int smem, int size_m, int size_n, int tilesize_n, int num_cus, int oversub_default = 1)
{
    int bps = rocm_coresident_blocks_per_cu(kernel, block_dim, smem);
#if EXL3_ROCM_ORDERED_EPILOGUE
    if (size_m <= 16)
    {
        static const int oversub_env = getenv("EXL3_ROCM_OVERSUB") ? atoi(getenv("EXL3_ROCM_OVERSUB")) : 0;
        const int oversub = oversub_env > 0 ? oversub_env : oversub_default;
        // the largest grid the ordered epilogue can serve for this shape (same test as the inner);
        // beyond it the kernel falls back to the locks and the hard bound applies
        int over = MAX(oversub, 1);
        while (over > 1 && !rocm_ordered_fits(num_cus * bps * over, size_n / tilesize_n, size_m, tilesize_n)) --over;
        if (over > 1 || rocm_ordered_fits(num_cus * bps, size_n / tilesize_n, size_m, tilesize_n)) bps *= over;
        // (if even 1x the bound does not fit the ordered buffer, the lock path runs at the bound)
    }
#endif
    return MAX(MIN(bps, 12), 1);
}
#endif

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    bool half_k
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) (half_k ? 1 : 0));
    mix((uint64_t) MIN(roundup_pow2(size_m), 16));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out,
    bool half_k
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb, half_k);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    // TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // Get SU, optionally
    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {
        // TORCH_CHECK_SHAPES(suh.value(), 0, A, 1, 1);
        A_had_ptr = (half*) OPTPTR(A_had);
        // TORCH_CHECK(A_had_ptr, "Must supply A_had with suh");
        // TORCH_CHECK_SHAPES_FULL(A_had.value(), A);
    }

    // Get SV, optionally
    const half* svh_ptr = (const half*) OPTPTR(svh);
    // if (svh_ptr)
        // TORCH_CHECK_SHAPES(svh.value(), 0, B, 1, 16);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch. 16 * K uint16 per tile for integer K; half-integer bitrates (K + 0.5, mul1 only) carry 16 * K + 8
    const int tile_u16 = B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1), "exl3_gemm: half-integer bitrates require the mul1 codebook");
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    // Experimental fused int8-activation GEMV path (EXL3_INT8_GEMV=1) for mul1 tensors. Rows are
    // processed as successive GEMV launches, so this is only sensible for small m (the reconstruct
    // threshold keeps m <= 144 in practice). Not graph-capturable yet; graphed callers fall through
    // to the regular kernel.
    if (mul1 && exl3_gemv_int8_enabled())
    {
        if (exl3_gemv_int8(A, B, C, suh, A_had, svh, stream, graph))
            return 0;
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

#if defined(USE_ROCM)
    // Split launch: input Hadamard (plain kernel) -> body (plain kernel) reading A_had as A.
    // Graph parameter recording: the had kernel exposes GP_gemm_A (arg 0) and GP_gemm_B_suh
    // (arg 2) and the body kernel the remaining ones; the site list below follows kernel order
    TORCH_CHECK(A_had_ptr && suh_ptr, "exl3_gemm (ROCm): suh and A_had are required");
    // Fused input Hadamard inside the body kernel (per-segment LDS rotation) when the segment's x
    // slice fits the LDS budget; otherwise the separate Hadamard kernel + body reading A_had.
    // The autotuned grid decides the segment length, so the bound is checked against the
    // smallest grid the autotuner may pick (see exl3_gemm_shape_compat / grid_cap below)
    // fused iff every compatible tile shape admits a grid within the device's block capacity
    bool fused_had = true;
    int fused_min_grid[EXL3_GEMM_NUM_SHAPES + 1] = {};
    for (int si = 1; si <= EXL3_GEMM_NUM_SHAPES; ++si)
    {
        if (!exl3_gemm_shape_compat(si, size_m, size_k, size_n, K)) continue;
        int g = exl3_rocm_fused_had_min_grid(size_m, size_k, size_n, exl3_gemm_tilesize_n_g[si]);
        fused_min_grid[si] = g;
    }
    // fused iff at least one compatible shape can satisfy its bound within its occupancy limit
    {
        bool any = false;
        for (int si = 1; si <= EXL3_GEMM_NUM_SHAPES; ++si)
        {
            if (!exl3_gemm_shape_compat(si, size_m, size_k, size_n, K) || fused_min_grid[si] < 0) continue;
            fp_exl3_gemm_kernel kf = get_gemm_kernel_ptr(K, si, c_fp32, cb, half_k);
            if (!kf) continue;
            int bps = rocm_gemm_blocks_per_cu((const void*) kf, exl3_gemm_blockdim_g[si], GEMM_DYN_SMEM, size_m, size_n, exl3_gemm_tilesize_n_g[si], num_sms);
            int tilesize_k = exl3_gemm_tilesize_k_g[si];
            int max_slices = MAX(size_k / tilesize_k * size_n / exl3_gemm_tilesize_n_g[si], 1);
            int cap = MIN(max_slices, num_sms * bps);
            if (fused_min_grid[si] <= cap) any = true;
        }
        fused_had = any;
    }
    if (force_shape_idx > 0 || force_num_sms > 0)
    {
        // benchmark overrides bypass the autotuner bound: fused only when the forced grid fits
        static const bool dbg_force_fused = getenv("EXL3_ROCM_FORCE_FUSED_HAD") != nullptr;
        fused_had = fused_had && dbg_force_fused && force_shape_idx > 0 && force_num_sms >= fused_min_grid[force_shape_idx];
    }
    const half* body_suh = nullptr;
    const half* A_body_ptr = A_ptr;
    if (fused_had)
    {
        body_suh = suh_ptr;
    }
    else
    {
        int total_warps = size_m * size_k / 128;
        int had_blocks = CEIL_DIVIDE(total_warps, 8);
        exl3_gemm_had_kernel<<<had_blocks, 256, 0, stream>>>(A_ptr, A_had_ptr, suh_ptr, size_k, total_warps);
        if (graph)
        {
            graph->record_param((void*) exl3_gemm_had_kernel, GP_gemm_A, 0);
            graph->record_param((void*) exl3_gemm_had_kernel, GP_gemm_A_had, 1);
            graph->record_param((void*) exl3_gemm_had_kernel, GP_gemm_B_suh, 2);
        }
        A_body_ptr = A_had_ptr;
    }
    void* kernelArgs[] =
    {
        (void*)& A_body_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& body_suh,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };
#else
    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };
#endif

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
#if defined(USE_ROCM)
            graph->record_param(kernel_ptr, fused_had ? GP_gemm_A : GP_gemm_A_had, 0);
#else
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
#endif
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    // QTIP-style GEMV path for small m (exl3_gemv_kernel.cuh). Same kernel arguments, so graph
    // recording is identical; falls through to the regular kernel when the heuristic declines
    if (force_shape_idx <= 0 && force_num_sms <= 0)
    {
        void* gemv_kernel = nullptr;
        if (exl3_gemv_try_launch
        (
            kernelArgs, size_m, size_k, size_n, K, half_k, cb, c_fp32,
            suh_ptr && A_had_ptr && svh_ptr,
            device, stream, &gemv_kernel, false
        ))
        {
            add_graph_args(gemv_kernel);
            cuda_check(cudaPeekAtLastError());
            return 90;
        }
    }

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb, half_k);
#if defined(USE_ROCM)
        if (fused_had) autotune_key ^= 0xF05EDA0Dull;
#endif
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, GEMM_DYN_SMEM, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        std::vector<CoopAutotuneCandidate> candidates;
        for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
        {
            if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

            fp_exl3_gemm_kernel candidate_kernel = get_gemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb, half_k);
            if (!candidate_kernel) continue;

            int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
            int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
            int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
            int grid_cap = num_sms;
#if defined(USE_ROCM)
            // The HIP inner is a latency-bound streaming kernel (no tensor-core pipeline), so it
            // profits from several resident blocks per CU; let the autotuner search up to the
            // MEASURED co-residency limit (the column lock deadlocks - and hangs the GPU - if any
            // block of the grid is not resident; the occupancy API is not that bound on gfx11)
            {
                int blocks_per_sm = rocm_gemm_blocks_per_cu
                (
                    (const void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx], GEMM_DYN_SMEM, size_m, size_n, tilesize_n, num_sms
                );
                grid_cap = num_sms * blocks_per_sm;
            }
#endif
            int max_candidate_sms = MAX(MIN(max_slices, grid_cap), 1);

            CoopAutotuneCandidate cand
            {
                (void*) candidate_kernel,
                exl3_gemm_blockdim_g[candidate_shape_idx],
                max_candidate_sms,
                1,
#if defined(USE_ROCM)
                num_sms,             // total_sms = CU count: the autotuner samples multiples of it
#else
                max_candidate_sms,
#endif
                candidate_shape_idx
            };
#if defined(USE_ROCM)
            if (fused_had)
            {
                // a shape whose smallest LDS-safe grid exceeds its occupancy-limited maximum is
                // not a candidate at all (the segment slice would overflow the LDS)
                if (fused_min_grid[candidate_shape_idx] > max_candidate_sms) continue;
                cand.min_num_sms = fused_min_grid[candidate_shape_idx];
            }
#endif
            candidates.push_back(cand);
        }
        TORCH_CHECK(!candidates.empty(), "exl3_gemm autotune: no compatible kernel shapes");

        tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, GEMM_DYN_SMEM, stream, (size_t) size_k * size_n);
        if (graph)
        add_graph_args((void*) tuned.kernel);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, half_k
    );
    if (!kernel) return 0;

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM_DYN_SMEM);
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    if (force_num_sms > 0)
    {
        // forced grid (benchmarks): reject an over-subscribed launch with a Python exception
        // instead of a GPU deadlock. EXL3_ROCM_UNSAFE_GRID=1 bypasses (co-residency experiments)
#if defined(USE_ROCM)
        static const bool unsafe = getenv("EXL3_ROCM_UNSAFE_GRID") != nullptr;
        int blocks_per_sm = rocm_gemm_blocks_per_cu((const void*) kernel, block_dim, GEMM_DYN_SMEM, size_m, size_n, exl3_gemm_tilesize_n_g[shape_idx], DevCtx::instance().get_num_sms(device));
#else
        const bool unsafe = false;
        int blocks_per_sm = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, (const void*) kernel, block_dim, GEMM_DYN_SMEM);
        cudaGetLastError();
#endif
        int max_blocks = blocks_per_sm * DevCtx::instance().get_num_sms(device);
        TORCH_CHECK(unsafe || num_sms <= max_blocks, "exl3_gemm: forced grid ", num_sms, " exceeds co-residency limit ", max_blocks, " for shape ", shape_idx);
    }
#if defined(USE_ROCM)
    cudaLaunchKernel((void*) kernel, dim3(num_sms), dim3(block_dim), kernelArgs, GEMM_DYN_SMEM, stream);
#else
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        GEMM_DYN_SMEM,
        stream
    );
#endif
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

#if defined(USE_ROCM)
int exl3_gemm2_gr
(
    const at::Tensor& A,
    const at::Tensor& B0, at::Tensor& C0, const at::Tensor& svh0, const at::Tensor& suh0,
    const at::Tensor& B1, at::Tensor& C1, const at::Tensor& svh1, const at::Tensor& suh1,
    bool mcg, bool mul1,
    int force_shape_idx, int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B0, 3); TORCH_CHECK_DIM(B1, 3);
    TORCH_CHECK_SHAPES(A, -1, B0, 0, 16);
    TORCH_CHECK_SHAPES(A, -1, B1, 0, 16);
    TORCH_CHECK_SHAPES(C0, -1, B0, 1, 16);
    TORCH_CHECK_SHAPES(C1, -1, B1, 1, 16);
    TORCH_CHECK(B0.size(2) == B1.size(2), "exl3_gemm2: both matrices must have the same bitrate");
    TORCH_CHECK_DTYPE(A, kHalf); TORCH_CHECK_DTYPE(B0, kShort);
    const bool c_fp32 = C0.dtype() == at::kFloat;
    TORCH_CHECK(C1.dtype() == C0.dtype(), "exl3_gemm2: C0 and C1 must have the same dtype");
    if (!c_fp32) { TORCH_CHECK_DTYPE(C0, kHalf); }
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1");

    int device; cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    const int tile_u16 = B0.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1), "exl3_gemm2: half-integer bitrates require the mul1 codebook");
    int cb = mcg ? 1 : (mul1 ? 2 : 0);

    int size_m = 1;
    for (int d = 0; d < A.dim() - 1; ++d) size_m *= A.size(d);
    TORCH_CHECK(size_m <= 16, "exl3_gemm2: size_m <= 16 only (decode path)");
    int size_k = A.size(-1);
    int size_n = B0.size(1) * 16;
    int size_n1 = B1.size(1) * 16;
    const int size_n_tot = size_n + size_n1;   // virtual width for tile counts / grid bounds

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B0_ptr = (const uint16_t*) B0.data_ptr();
    const uint16_t* B1_ptr = (const uint16_t*) B1.data_ptr();
    void* C0_ptr = C0.data_ptr(); void* C1_ptr = C1.data_ptr();
    const half* suh_ptr = (const half*) suh0.data_ptr();
    const half* suh1_ptr = (const half*) suh1.data_ptr();
    const half* svh0_ptr = (const half*) svh0.data_ptr();
    const half* svh1_ptr = (const half*) svh1.data_ptr();
    half* A_had_dummy = nullptr;

    // always the fused input Hadamard: with 2x the column tiles every shape's minimum LDS-safe
    // grid halves relative to the single GEMM, and the grid bound below is checked against it
    void* kernelArgs[] =
    {
        (void*)& A_ptr, (void*)& B0_ptr, (void*)& C0_ptr, (void*)& size_m, (void*)& size_k, (void*)& size_n,
        (void*)& locks, (void*)& suh_ptr, (void*)& A_had_dummy, (void*)& svh0_ptr,
        (void*)& B1_ptr, (void*)& C1_ptr, (void*)& svh1_ptr, (void*)& suh1_ptr, (void*)& size_n1
    };
    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm2_C1, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    int fused_min_grid[EXL3_GEMM_NUM_SHAPES + 1] = {};
    for (int si = 1; si <= EXL3_GEMM_NUM_SHAPES; ++si)
    {
        if (!exl3_gemm_shape_compat(si, size_m, size_k, size_n, K)) continue;
        // the virtual matrix is size_n_tot wide
        fused_min_grid[si] = exl3_rocm_fused_had_min_grid(size_m, size_k, size_n_tot, exl3_gemm_tilesize_n_g[si]);
    }

    if (force_shape_idx <= 0 && force_num_sms <= 0)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb, half_k) ^ 0x6E3322ull ^ ((uint64_t) size_n1 * 0x9E3779B97F4A7C15ull);
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, GEMM_DYN_SMEM, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        std::vector<CoopAutotuneCandidate> candidates;
        for (int si = 1; si <= EXL3_GEMM_NUM_SHAPES; ++si)
        {
            if (!exl3_gemm_shape_compat(si, size_m, size_k, size_n, K) || !exl3_gemm_shape_compat(si, size_m, size_k, size_n1, K) || fused_min_grid[si] < 0) continue;
            fp_exl3_gemm2_kernel kf = get_gemm2_kernel_ptr(K, si, cb, half_k, c_fp32);
            if (!kf) continue;
            int tilesize_n = exl3_gemm_tilesize_n_g[si];
            int max_slices = MAX(size_k / exl3_gemm_tilesize_k_g[si] * (size_n_tot) / tilesize_n, 1);
            int bps = rocm_gemm_blocks_per_cu((const void*) kf, exl3_gemm_blockdim_g[si], GEMM_DYN_SMEM, size_m, size_n_tot, tilesize_n, num_sms, 2);
            int max_candidate_sms = MAX(MIN(max_slices, num_sms * bps), 1);
            if (fused_min_grid[si] > max_candidate_sms) continue;
            CoopAutotuneCandidate cand { (void*) kf, exl3_gemm_blockdim_g[si], max_candidate_sms, 1, num_sms, si };
            cand.min_num_sms = fused_min_grid[si];
            candidates.push_back(cand);
        }
        if (candidates.empty()) return -1;
        tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, GEMM_DYN_SMEM, stream, (size_t) size_k * size_n * 2);
        if (graph) add_graph_args((void*) tuned.kernel);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    // forced (benchmarks)
    int si = force_shape_idx > 0 ? force_shape_idx : 3;
    TORCH_CHECK(exl3_gemm_shape_compat(si, size_m, size_k, size_n, K), "exl3_gemm2: forced shape incompatible");
    fp_exl3_gemm2_kernel kernel = get_gemm2_kernel_ptr(K, si, cb, half_k, c_fp32);
    int tilesize_n = exl3_gemm_tilesize_n_g[si];
    int grid = force_num_sms > 0 ? force_num_sms : num_sms;
    int max_slices = MAX(size_k / exl3_gemm_tilesize_k_g[si] * (size_n_tot) / tilesize_n, 1);
    grid = MAX(MIN(grid, max_slices), 1);
    TORCH_CHECK(fused_min_grid[si] >= 1 && grid >= fused_min_grid[si], "exl3_gemm2: forced grid ", grid, " below the fused-had minimum ", fused_min_grid[si]);
    static const bool unsafe = getenv("EXL3_ROCM_UNSAFE_GRID") != nullptr;
    int bps = rocm_gemm_blocks_per_cu((const void*) kernel, exl3_gemm_blockdim_g[si], GEMM_DYN_SMEM, size_m, size_n_tot, tilesize_n, num_sms, 2);
    TORCH_CHECK(unsafe || grid <= bps * num_sms, "exl3_gemm2: forced grid ", grid, " exceeds co-residency limit ", bps * num_sms, " for shape ", si);
    cudaLaunchKernel((void*) kernel, dim3(grid), dim3(exl3_gemm_blockdim_g[si]), kernelArgs, GEMM_DYN_SMEM, stream);
    add_graph_args((void*) kernel);
    cuda_check(cudaPeekAtLastError());
    return si;
}

int exl3_gemm2
(
    const at::Tensor& A,
    const at::Tensor& B0, at::Tensor& C0, const at::Tensor& svh0, const at::Tensor& suh0,
    const at::Tensor& B1, at::Tensor& C1, const at::Tensor& svh1, const at::Tensor& suh1,
    bool mcg, bool mul1,
    int force_shape_idx, int force_num_sms
)
{
    return exl3_gemm2_gr(A, B0, C0, svh0, suh0, B1, C1, svh1, suh1, mcg, mul1, force_shape_idx, force_num_sms, nullptr);
}

// debug: copy the per-block probe area (see EXL3_ROCM_PROBE in the inner) into a tensor
void exl3_rocm_probe_read(at::Tensor out)
{
    int device; cudaGetDevice(&device);
    int* locks = DevCtx::instance().get_locks(device);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data_ptr(), locks + 65536, out.numel() * out.element_size(), cudaMemcpyDeviceToDevice);
    cudaMemset(locks + 65536, 0, out.numel() * out.element_size());
    cudaDeviceSynchronize();
}
#endif

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    return exl3_gemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        force_shape_idx,
        mcg,
        mul1,
        force_num_sms,
        nullptr
    );
}

/*
EXL3 batched/multi-matrix matmul.

This is not a conventional batched A @ B. B, suh and svh are CUDA int64
tensors containing device addresses (one address per quantized matrix), rather
than the matrix data themselves. Entry q of each table describes one linear:

    B[q]   -> EXL3 trellis, logically (k / 16, n / 16, 16 * K) uint16
    suh[q] -> packed input scales/flips, logically (k / 16) float16
    svh[q] -> packed output scales/flips, logically (n / 16) float16

A is contiguous float16 [a_batches, m, k], C is contiguous float16 or
float32 [c_batches, m, n], and A_had is float16 scratch with room for every
active matrix. The kernel applies the input Hadamard transform into A_had,
performs the selected EXL3 matmul, then applies the output transform.

The active matrix/output slot j selects q = indices[j] when indices is given,
or q = j otherwise. This supports the following modes:

- Multiple inputs and outputs: A[j] @ B[q] -> C[j].
- One input, multiple outputs: when a_batches == 1, A[0] is broadcast and
  transformed separately for each selected B[q], producing C[j]. This is used
  for e.g. fused gate/up projections and MoE expert fan-out.
- Indexed matrices: indices is a contiguous int64 [*, num_indices] tensor;
  the kernel reads its first num_indices entries as q values. Negative indices
  skip that slot.
- Weighted MoE reduction: weights is a float16 tensor parallel to indices.
  Each transformed result is multiplied by weights[j], then all active C[j]
  are summed into C[0]. C therefore also serves as per-expert scratch; only
  C[0] is the reduced result.
- Expert-range filtering: with min_index >= 0, selections outside
  [min_index, max_index) are removed and retained indices are rebased by
  min_index. This allows B/suh/svh to be local pointer tables for an expert
  shard. At num_tokens == 1 the retained indices (and their weights) are
  compacted; at num_tokens > 1 out-of-range slots are instead masked to -1 in
  place, preserving the per-token slot groups the final reduction depends on
  (and, with bszm_in > 1, the slot -> input-row correspondence).

Without weights, every active C[j] is a separate output. The active slot count
is max(a_batches, c_batches), capped to num_indices when indices is present.

Limitations: k must be divisible by 16 and n by 128. Range filtering supports
at most 128 slots (the kernel's index-compaction capacity).
*/

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K_,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    // num_tokens > 1 with expert-range filtering (min_index >= 0) uses position-preserving
    // masking in the kernel (out-of-range slots marked -1 in place) instead of index
    // compaction, so the grouped reduction's fixed per-token slot runs stay intact

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(svh, kLong);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(C, 3);

    TORCH_CHECK_SHAPES(A, 1, C, 1, 1);
    if (!had_src_list) TORCH_CHECK_SHAPES(B, 0, suh, 0, 1);   // sliced mode: suh is per source
    TORCH_CHECK_SHAPES(B, 0, svh, 0, 1);

    int bsz = A.size(1);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);

    // Per-matrix output widths/pointers (uniform-width callers pass neither): C then only
    // provides the dtype and the max width (locks/shape sizing); outputs go to c_ptrs
    const int* size_n_list_ptr = nullptr;
    void** c_list_ptr = nullptr;
    if (size_n_list)
    {
        TORCH_CHECK(c_ptrs, "exl3_mgemm: size_n_list requires c_ptrs");
        TORCH_CHECK_DTYPE(size_n_list.value(), kInt);
        TORCH_CHECK_DTYPE(c_ptrs.value(), kLong);
        TORCH_CHECK(num_tokens == 1 && min_index < 0 && !weights,
                    "exl3_mgemm: per-matrix widths incompatible with multi-token/filtering/weights");
        size_n_list_ptr = (const int*) size_n_list.value().data_ptr();
        c_list_ptr = (void**) c_ptrs.value().data_ptr();
        bszm_out = (int) c_ptrs.value().size(0);
    }
    int bszm = MAX(bszm_in, bszm_out);

    // Sliced mode: the entries are equal-width column slices of num_had_src source matrices,
    // scheduled as independent z-groups so unequal matrices (e.g. Q vs K/V) can't leave groups
    // idle. B/svh/c_ptrs entries are pre-offset to the slice's first column, size_n_list holds
    // the slice width, n_stride_list the source's full width (the row stride of B and C), and
    // had_src_list maps each slice to its source: suh and the A_had slabs are per source
    const int* n_stride_list_ptr = nullptr;
    const int* had_src_list_ptr = nullptr;
    if (had_src_list)
    {
        TORCH_CHECK(size_n_list && c_ptrs && n_stride_list,
                    "exl3_mgemm: sliced mode requires size_n_list, c_ptrs and n_stride_list");
        TORCH_CHECK(bszm_in == 1 && !indices && !weights && num_tokens == 1 && min_index < 0,
                    "exl3_mgemm: sliced mode is single-input, unfiltered and single-token");
        TORCH_CHECK_DTYPE(had_src_list.value(), kInt);
        TORCH_CHECK_DTYPE(n_stride_list.value(), kInt);
        TORCH_CHECK(had_src_list.value().size(0) >= bszm_out && n_stride_list.value().size(0) >= bszm_out,
                    "exl3_mgemm: had_src_list / n_stride_list must have one entry per slice");
        TORCH_CHECK(num_had_src > 0 && suh.size(0) >= num_had_src,
                    "exl3_mgemm: sliced mode needs one suh entry per source");
        n_stride_list_ptr = (const int*) n_stride_list.value().data_ptr();
        had_src_list_ptr = (const int*) had_src_list.value().data_ptr();
    }
    else
    {
        TORCH_CHECK(!n_stride_list, "exl3_mgemm: n_stride_list requires had_src_list");
    }

    // The kernel writes one hadamard-transformed input slab PER MATRIX (A_had + j * m * k), or
    // per source in sliced mode; an undersized scratch is silent OOB corruption (found the hard way)
    int64_t had_slabs = had_src_list ? num_had_src : bszm;
    TORCH_CHECK(A_had.numel() >= had_slabs * A.size(1) * A.size(2),
                "exl3_mgemm: A_had must hold bszm * m * k elements");

    const int64_t* indices_ptr = (const int64_t*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    if (indices)
    {
        TORCH_CHECK_DIM(indices.value(), 2);
        int num_indices = indices.value().size(1);
        TORCH_CHECK(num_indices <= bszm_in || num_indices <= bszm_out, "mgemm: too many indices for tensor batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
    }

    if (weights)
    {
        TORCH_CHECK_DIM(weights.value(), 2);
    }

    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    const half* A_ptr = (const half*) A.data_ptr();
    const uintptr_t* B_ptr_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* A_had_ptr = (const half*) A_had.data_ptr();
    const uintptr_t* suh_ptr_ptr = (const uintptr_t*) suh.data_ptr();
    const uintptr_t* svh_ptr_ptr = (const uintptr_t*) svh.data_ptr();

    // Select kernel. K_ is the bitrate (integer or half-integer, see bits_k.cuh)
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    const bool half_k = bk.half;
    TORCH_CHECK(!half_k || mul1, "exl3_mgemm: half-integer bitrates require the mul1 codebook");

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel;
    int concurrency;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr_ptr,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list_ptr,
        (void*)& c_list_ptr,
        (void*)& n_stride_list_ptr,
        (void*)& had_src_list_ptr,
        (void*)& num_had_src
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_mgemm_A, 0);
            graph->record_param(kernel_ptr, GP_mgemm_C, 2);
            graph->record_param(kernel_ptr, GP_mgemm_indices, 10);
            graph->record_param(kernel_ptr, GP_mgemm_weights, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = mgemm_autotune_hash
        (
            size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out, half_k
        );
        if (had_src_list) autotune_key ^= 0x9e3779b97f4a7c15ull;   // sliced launches tune separately

        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

                fp_exl3_mgemm_kernel candidate_kernel = get_mgemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb, half_k);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, total_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    bszm,
                    total_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_mgemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n * bszm);
            add_graph_args((void*) tuned.kernel);

            // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, tuned.tag, tuned.block_dim, tuned.num_sms, tuned.concurrency);

            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out, half_k
    );
    int tilesize_k = exl3_gemm_tilesize_k_g[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n_g[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    concurrency = MIN(total_sms / num_sms, bszm);

    // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, shape_idx, block_dim, num_sms, concurrency);

    // Launch bigger grid if possible
    dim3 block_grid(num_sms, 1, concurrency);

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set[device].insert((void*) kernel);
    }

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    float K_,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src
)
{
    return exl3_mgemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        indices,
        weights,
        K_,
        force_shape_idx,
        mcg_mult,
        mul1_mult,
        min_index,
        max_index,
        force_num_sms,
        nullptr,
        num_tokens,
        size_n_list,
        c_ptrs,
        n_stride_list,
        had_src_list,
        num_had_src
    );
}
