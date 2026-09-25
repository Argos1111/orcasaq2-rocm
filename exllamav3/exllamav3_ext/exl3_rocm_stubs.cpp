// HIP definitions for the two warp-mma GEMV engine launchers, excluded from the
// ROCm build (build_config.py). Both always decline; the split-K streaming inner
// covers those shapes instead.

#include <ATen/Tensor.h>
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_gemv_int8.cuh"
#include "graph.cuh"

#if defined(USE_ROCM)

// always declines; A-hadamard staging stays the caller's responsibility
bool exl3_gemv_try_launch
(
    void**,
    int, int, int, int,
    bool,
    int,
    bool, bool,
    int,
    cudaStream_t,
    void**,
    bool
)
{
    return false;
}

void* exl3_gemv_select_kernel_half(int, bool, int, int, bool) { return nullptr; }

void exl3_gemv
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    bool,
    bool
)
{
    TORCH_CHECK(false, "exl3_gemv: direct GEMV entry point is not built on ROCm (the regular exl3_gemm kernel covers these shapes)");
}

bool exl3_gemv_int8_enabled() { return false; }

bool exl3_gemv_int8
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    cudaStream_t,
    Graph*
)
{
    return false;
}

int exl3_gemv_int8_max_k(int) { return 0; }

// hgemm_f16acc.cu is the sm_120 PTX tensor-core GEMM (GeForce fp16-accumulator path); not
// ported. hgemm.cu falls back to rocBLAS when hgemm_f16acc_try returns false. hgemm_recon
// (defined there upstream) reduces to plain hgemm under the stub.
#include "hgemm.cuh"

bool hgemm_f16acc_try(const at::Tensor&, const at::Tensor&, at::Tensor&) { return false; }

void hgemm_f16acc(at::Tensor, at::Tensor, at::Tensor)
{
    TORCH_CHECK(false, "hgemm_f16acc: not built on ROCm (the fp32-accumulator path covers it)");
}

int hgemm_f16acc_status(int) { return 0; }

void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    hgemm(a, b, c);
}

// Fused coop MoE kernel (exl3_moe_coop.cu) is not ported: it builds on the warp-matrix GEMV
// engines. run_bszN routes through exl3_moe_coop_run, which callers reach only when
// InferParams.use_mgemm() is true (declined on HIP), so the run stub fails loudly if reached.
// prepare must return rather than throw: the BC_BlockSparseMLP constructor calls it on every
// platform, and its result is never dereferenced when the coop path is declined.
#include "quant/exl3_moe_coop.cuh"

MoeCoopParams exl3_moe_coop_prepare
(
    int,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    float, float, float,
    bool, bool,
    int, float,
    bool,
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,      // had_g, had_u, gu_g, gu_u
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,      // act_out, d_out, ctr, out
    const c10::optional<at::Tensor>&,                        // sh_gate_w
    float&, float&, int&
)
{
    return MoeCoopParams{};
}

void exl3_moe_coop_run
(
    MoeCoopParams, float, float, int,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&
)
{
    TORCH_CHECK(false, "exl3_moe_coop_run: coop MoE kernel is not built on ROCm (use_mgemm declines the path)");
}

void exl3_moe_coop_launch(const MoeCoopParams&, float, float, int, int, cudaStream_t)
{
    TORCH_CHECK(false, "exl3_moe_coop_launch: coop MoE kernel is not built on ROCm");
}

void exl3_moe_coop
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int, int, int,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    float, float, float,
    bool, bool,
    int, float,
    bool,
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,      // had_g, had_u, gu_g, gu_u
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,      // act_out, d_out, ctr, out
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&
)
{
    TORCH_CHECK(false, "exl3_moe_coop: coop MoE kernel is not built on ROCm");
}


// det_gemm.cuh (mma.sync / cp.async / ldmatrix PTX) has no HIP port: the deterministic int8
// tensor-core routing GEMM and the tiled GatedResidual mix are excluded (build_config.py).
// Callers fall back to their cuBLAS / eager paths when these decline
#include "routing.cuh"
#include "hc_mix.cuh"

bool routing_gemm_det_fits(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) { return false; }
void routing_gemm_det_(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, cudaStream_t)
{
    TORCH_CHECK(false, "routing_gemm_det: not built on ROCm");
}
void routing_gemm_det(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor)
{
    TORCH_CHECK(false, "routing_gemm_det: not built on ROCm");
}
void det_quant_weight(const at::Tensor&, at::Tensor, at::Tensor)
{
    TORCH_CHECK(false, "det_quant_weight: not built on ROCm");
}
void det_math_test(const at::Tensor&, at::Tensor)
{
    TORCH_CHECK(false, "det_math_test: not built on ROCm");
}
int gr_mix_tiled_slices(int, int, int) { return 0; }
void gr_mix_tiled
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, double, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    c10::optional<at::Tensor>, at::Tensor
)
{
    TORCH_CHECK(false, "gr_mix_tiled: not built on ROCm");
}

#endif
