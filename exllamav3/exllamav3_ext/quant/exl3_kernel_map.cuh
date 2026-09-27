#pragma once

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits, bool multi);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits);

// bits: integer part of the bitrate; half: bitrate is bits + 0.5 (mul1 codebook only, 16 * bits + 8 uint16 per tile)
#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool half_k, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

#define EXL3_MGEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t** __restrict__ B_list, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half** __restrict__ suh_list, \
    half* __restrict__ A_had, \
    const half** __restrict__ svh_list, \
    int64_t* B_indices, \
    half* B_weights, \
    const int bszm_in, \
    const int bszm_out, \
    const int min_index, \
    const int max_index, \
    const int num_tokens, \
    const int* __restrict__ size_n_list, \
    void** __restrict__ C_list, \
    const int* __restrict__ n_stride_list, \
    const int* __restrict__ had_src_list, \
    const int num_had_src

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);
typedef void (*fp_exl3_gemm2_kernel) (EXL3_GEMM_ARGS, const uint16_t* __restrict__, void* __restrict__, const half* __restrict__, const half* __restrict__, const int);
typedef void (*fp_exl3_mgemm_kernel) (EXL3_MGEMM_ARGS);

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

// ROCm: the single-GEMM tables hold the plain-launch body kernel (exl3_gemm_body_kernel); the
// input Hadamard runs as a separate kernel. See exl3_gemm_kernel.cuh
#if defined(USE_ROCM)
#define EXL3_GEMM_SINGLE_KERNEL exl3_gemm_body_kernel
#else
#define EXL3_GEMM_SINGLE_KERNEL exl3_gemm_kernel
#endif

// Shape 1 not currently used anywhere
#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

// Half-integer bitrates (bits + 0.5), mul1 codebook, single GEMM only (no mgemm bundles yet)
#define EXL3_GEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_1>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_2>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_3>, \
    EXL3_GEMM_SINGLE_KERNEL<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_4>

#define EXL3_MGEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_3>, \
    exl3_mgemm_kernel<_bits, true, _c_fp32, 2, EXL3_GEMM_SHAPE_4>

#if defined(USE_ROCM)
#define EXL3_GEMM2_KERNEL_INSTANCES(_bits, _half_k, _c_fp32, cb) \
    nullptr, \
    exl3_gemm2_body_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm2_body_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm2_body_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_gemm2_body_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_4>
#define EXL3_GEMM2_INSTANCES_H(K) \
    EXL3_GEMMM_TABLES(h##K, K, true) \
    fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp16_h##K[] = { EXL3_GEMM2_KERNEL_INSTANCES(K, true, false, 2) }; \
    fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp32_h##K[] = { EXL3_GEMM2_KERNEL_INSTANCES(K, true, true, 2) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp16_h##K[] = { EXL3_GEMMD_KERNEL_INSTANCES(K, true, false, 2) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp32_h##K[] = { EXL3_GEMMD_KERNEL_INSTANCES(K, true, true, 2) };
#define EXL3_GEMM2_EXTERNS_H(K) \
    EXL3_GEMMM_EXTERNS(h##K) \
    extern fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp16_h##K[]; \
    extern fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp32_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp16_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp32_h##K[];
// decode (lane-only) instances, same signature as the body kernel
#define EXL3_GEMMD_KERNEL_INSTANCES(_bits, _half_k, _c_fp32, cb) \
    nullptr, \
    exl3_gemm_decode_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_gemm_decode_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_gemm_decode_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_gemm_decode_kernel<_bits, _half_k, _c_fp32, cb, EXL3_GEMM_SHAPE_4>
// multi-row decode instances (mul1 only -> cb 2 tables; MR 2 / 4 / 8)
#define EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, _c_fp32, MR) \
    nullptr, \
    exl3_gemm_mrows_kernel<_bits, _half_k, _c_fp32, 2, EXL3_GEMM_SHAPE_1, MR>, \
    exl3_gemm_mrows_kernel<_bits, _half_k, _c_fp32, 2, EXL3_GEMM_SHAPE_2, MR>, \
    exl3_gemm_mrows_kernel<_bits, _half_k, _c_fp32, 2, EXL3_GEMM_SHAPE_3, MR>, \
    exl3_gemm_mrows_kernel<_bits, _half_k, _c_fp32, 2, EXL3_GEMM_SHAPE_4, MR>
#define EXL3_GEMMM_TABLES(sfx, _bits, _half_k) \
    fp_exl3_gemm_kernel tfp_exl3_gemmm2_kernel_fp16_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, false, 2) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm4_kernel_fp16_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, false, 4) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm6_kernel_fp16_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, false, 6) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm8_kernel_fp16_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, false, 8) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm2_kernel_fp32_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, true, 2) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm4_kernel_fp32_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, true, 4) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm6_kernel_fp32_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, true, 6) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmm8_kernel_fp32_##sfx[] = { EXL3_GEMMM_KERNEL_INSTANCES(_bits, _half_k, true, 8) };
#define EXL3_GEMMM_EXTERNS(sfx) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmm2_kernel_fp16_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm4_kernel_fp16_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm6_kernel_fp16_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm8_kernel_fp16_##sfx[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmm2_kernel_fp32_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm4_kernel_fp32_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm6_kernel_fp32_##sfx[]; extern fp_exl3_gemm_kernel tfp_exl3_gemmm8_kernel_fp32_##sfx[];
// integer K: one gemm2 table per (K, cb) alongside the regular ones
#define EXL3_GEMMM_INSTANCES_CB_2(K) EXL3_GEMMM_TABLES(b##K, K, false)
#define EXL3_GEMMM_INSTANCES_CB_1(K)
#define EXL3_GEMMM_INSTANCES_CB_0(K)
#define EXL3_GEMM2_INSTANCES_CB(K, cb) \
    EXL3_GEMMM_INSTANCES_CB_##cb(K) \
    fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp16_b##K##_cb##cb[] = { EXL3_GEMM2_KERNEL_INSTANCES(K, false, false, cb) }; \
    fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp32_b##K##_cb##cb[] = { EXL3_GEMM2_KERNEL_INSTANCES(K, false, true, cb) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp16_b##K##_cb##cb[] = { EXL3_GEMMD_KERNEL_INSTANCES(K, false, false, cb) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp32_b##K##_cb##cb[] = { EXL3_GEMMD_KERNEL_INSTANCES(K, false, true, cb) };
#define EXL3_GEMMM_EXTERNS_CB_2(K) EXL3_GEMMM_EXTERNS(b##K)
#define EXL3_GEMMM_EXTERNS_CB_1(K)
#define EXL3_GEMMM_EXTERNS_CB_0(K)
#define EXL3_GEMM2_EXTERNS_CB(K, cb) \
    EXL3_GEMMM_EXTERNS_CB_##cb(K) \
    extern fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp16_b##K##_cb##cb[]; \
    extern fp_exl3_gemm2_kernel tfp_exl3_gemm2_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp16_b##K##_cb##cb[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemmd_kernel_fp32_b##K##_cb##cb[];
#else
#define EXL3_GEMM2_INSTANCES_H(K)
#define EXL3_GEMM2_EXTERNS_H(K)
#define EXL3_GEMM2_INSTANCES_CB(K, cb)
#define EXL3_GEMM2_EXTERNS_CB(K, cb)
#endif

#define EXL3_KERNEL_INSTANCES_H(K) \
    EXL3_GEMM2_INSTANCES_H(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, false) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, false) };

#define EXL3_KERNEL_EXTERNS_H(K) \
    EXL3_GEMM2_EXTERNS_H(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[];

#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_1>, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_2>, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_3>, \
    exl3_mgemm_kernel<_bits, false, _c_fp32, cb, EXL3_GEMM_SHAPE_4>

#define EXL3_GEMM_BASE_THREADS 256

// Instance arrays are indexed by shape and defined per (K, cb) so each codebook compiles as a separate
// translation unit (see comp_units/exl3_comp_unit_K_cbX.cu)

#define EXL3_KERNEL_EXTERNS_CB(K, cb) \
    EXL3_GEMM2_EXTERNS_CB(K, cb) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[]; \

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    EXL3_KERNEL_EXTERNS_CB(K, 0) \
    EXL3_KERNEL_EXTERNS_CB(K, 1) \
    EXL3_KERNEL_EXTERNS_CB(K, 2) \

#define EXL3_KERNEL_INSTANCES_CB(K, cb) \
    EXL3_GEMM2_INSTANCES_CB(K, cb) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, cb) \
    };

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int bits,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const bool half_k = false
);

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const int bszm_in,
    const int bszm_out,
    const bool half_k = false
);

fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
#if defined(USE_ROCM)
fp_exl3_gemm2_kernel get_gemm2_kernel_ptr(int K, int shape_idx, int cb, bool half_k = false, bool c_fp32 = false);
fp_exl3_gemm_kernel get_gemmd_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);   // decode (lane-only) body
fp_exl3_gemm_kernel get_gemmm_kernel_ptr(int K, int shape_idx, bool c_fp32, int mr, bool half_k = false);   // multi-row decode body (mul1; mr 2/4/8), nullptr if none
#endif
fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
