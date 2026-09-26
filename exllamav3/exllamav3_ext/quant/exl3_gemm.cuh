#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

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
);

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
);

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
    float K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    // Sliced mode (see exl3_mgemm_gr): per-entry full row width of the slice's matrix, per-entry
    // source matrix index (suh and A_had are then per source), and the number of sources
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);

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
    float K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);

#if defined(USE_ROCM)
void exl3_rocm_probe_read(at::Tensor out);

// Dual GEMM (gate/up fusion): C0 = A @ B0, C1 = A @ B1 for two matrices of identical (k, n, K,
// codebook) in one launch; each with its own suh/svh. size_m <= 16 (decode). Graph params:
// GP_gemm_A (0), GP_gemm_C (2), GP_gemm2_C1 (11)
int exl3_gemm2_gr
(
    const at::Tensor& A,
    const at::Tensor& B0, at::Tensor& C0, const at::Tensor& svh0, const at::Tensor& suh0,
    const at::Tensor& B1, at::Tensor& C1, const at::Tensor& svh1, const at::Tensor& suh1,
    bool mcg, bool mul1,
    int force_shape_idx, int force_num_sms,
    Graph* graph
);
int exl3_gemm2
(
    const at::Tensor& A,
    const at::Tensor& B0, at::Tensor& C0, const at::Tensor& svh0, const at::Tensor& suh0,
    const at::Tensor& B1, at::Tensor& C1, const at::Tensor& svh1, const at::Tensor& suh1,
    bool mcg, bool mul1,
    int force_shape_idx, int force_num_sms
);
#endif
