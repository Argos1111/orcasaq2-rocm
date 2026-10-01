# OrcaSAQ2-27B on RDNA3 / RDNA4 — what was measured and how the kernels work

All numbers: single stream, greedy, Q8 KV cache (`cache_mode: "8,8"`), chunk size 2048,
128 generated tokens after the prompt, measured with `bench_ctx.py` / `bench_decode.py`
in this directory. Decode numbers vary by ±0.5 tok/s between runs on the same binary.

## Results

| context | RX 7900 XTX prefill | decode | decode + MTP (draft 3) | R9700 prefill | decode | decode + MTP |
|---:|---:|---:|---:|---:|---:|---:|
| 8k   | 7.2 s (1131 tok/s) | 38.0 tok/s | **51.9** | 6.0 s (1363) | 36.2 | **49.8** |
| 32k  | 23.9 s (1373) | 30.8 | **45.3** | 18.1 s (1810) | 29.9 | **49.8** |
| 192k | 294 s (669) | 13.5 | **21.3** | 209 s (940) | 14.6 | **22.1** |

VRAM: 17 GiB after load on the XTX, 18.7 GiB at 192k context, 20.2 GiB with the MTP head
and its cache. 192k fits on both cards; ~256k is the ceiling on 24 GB.

Where the time goes for one decoded token (XTX, 8k, no MTP, 24.5 ms):

- EXL3 GEMMs 19.7 ms (3 bpw ~14, 3.5 bpw ~4, 4 bpw ~1, lm_head 6 bpw 1.1) — 9.77 GB of weights
  read per token; the 3/3.5 bpw kernels reach ~600 GB/s, 4 bpw ~850 GB/s of the 960 GB/s bus
- kernel launch gaps 3.8 ms (777 launches × ~3.6 µs, close to the HIP graph floor of ~2.9 µs)
- GatedDeltaNet small kernels 1.25 ms, attention 1.0 ms, norms 0.4 ms

Decode falls with context because the 16 full-attention layers re-read the whole Q8 KV
cache every token (6 GB at 192k); the 48 GatedDeltaNet layers are context-free. MTP amortises
that read over 2–3 tokens, which is why its gain grows with context.

## MTP speculative decoding

The checkpoint's 1-layer MTP head drafts k tokens; the target verifies k+1 rows in one forward.
On this fork `draft_num_tokens: 3` is the measured optimum (accepted 86 of 128 drafted tokens;
1 → 45.0, 2 → 47.4, **3 → 51.9**, 4 → 49.6 tok/s at 8k). This needed a GEMM path that decodes
each weight once and applies it to all rows (`exl3_gemm_mrows_kernel`); the original per-row
lane tier made a 5-row verify cost 3.5× a single row and MTP was slower than no MTP.
M=2/3/5 rows now cost 1.2× / 1.7× / 2.1× of M=1 (a tensor-core GEMM would be ~1.1×).

## How the kernels differ from upstream

Upstream exllamav3's EXL3 GEMM decodes trellis weights and multiplies with `mma.sync`
tensor-core instructions. RDNA3/RDNA4 have no equivalent for this data layout, so the fork
runs the whole thing on the vector ALUs:

- **Lane tier**: 8 lanes own one 16×16 subtile; each lane decodes its 3–4 words of trellis code
  with `v_alignbit_b32` → `v_mad_u32_u16`/`v_mad_u16` (the mul1 codebook's affine map in two
  full-rate 16-bit multiplies) → `v_sad_u8`/`v_sad_hi_u8` (byte sums packed as fp16 pairs) →
  `v_dot2_f32_f16`. 4.5 VALU ops per weight, issued as lock-step groups of 8 codes so the
  dependent chain latency is hidden (~85% VALU utilisation).
- **Decode-only kernel**: the prefill (warp/dq) tier lives in a separate kernel instance so the
  decode kernel needs 95 VGPRs instead of 113 → 4 blocks per CU instead of 3 (+1.7 tok/s).
- **Fused input Hadamard**: the EXL3 input rotation runs inside the GEMM on an LDS slice of x
  instead of as a separate kernel (−400 launches per token).
- **Dual GEMM**: gate/up (and GatedDeltaNet qkv/z) projections in one launch.
- **Fused epilogues**: residual add + the next layer's RMSNorm run inside the sublayer graph;
  the GatedDeltaNet b/a GEMV, gating, conv1d update and bf16 cast are one kernel.

### Safety

The GEMM grid is a plain launch (cooperative launch costs 20 µs per kernel on HIP). Blocks of
one output column used to wait for each other through a lock, which requires every block of the
grid to be resident at once — and `hipOccupancyMaxActiveBlocksPerMultiprocessor` is not that
bound on gfx11 (measured both over- and under-reporting). A mis-estimate deadlocks the GPU and,
on this machine, froze the host (PCIe AER + IOMMU timeout). Two layers of defence are in place:

1. **No inter-block waiting**: each block stores its fp32 partial, takes a per-column ticket, and
   the last arriver sums the partials in a fixed order. Deterministic — every output is
   bit-identical run to run — and no co-residency requirement.
2. **Measured co-residency**: for the remaining lock path (prefill, >16 rows) and as the
   autotuner's grid bound, a probe kernel with the target's register/LDS footprint measures how
   many blocks per CU are actually resident. Results are cached on disk per architecture.

A debug build (`-DEXL3_ROCM_BOUNDS`) turns out-of-range tile reads and lock waits into a GPU
trap that ends the process; `check_device.sh` uses it for the first run on a new GPU.

## Environment variables

| variable | effect |
|---|---|
| `HIP_VISIBLE_DEVICES=n` | select the GPU (one at a time) |
| `PYTORCH_ROCM_ARCH="gfx1100;gfx1201"` | build targets (set before `uv sync`) |
| `EXL3_ROCM_NO_GEMM2=1` | disable the dual gate/up and qkv/z GEMM |
| `EXL3_ROCM_NO_MROWS=1` | disable the multi-row decode tier (M=2..8) |
| `EXL3_NO_NORM_FUSE=1` | disable the residual + next-norm fusion |
| `EXL3_ROCM_HGEMM_FP32=1` | hipBLAS fp32-output GEMM in prefill (6× slower; the default computes in fp16 and widens) |
| `EXL3_ROCM_CORESIDENCY_VERBOSE=1` | print the measured blocks/CU per kernel |
| `EXL3_ROCM_UNSAFE_GRID=1` | let forced grids exceed the co-residency bound (experiments only) |
| `EXL3_ROCM_ALLOW_ANY_ARCH=1` | load on an architecture the gate refuses (CDNA, RDNA2) — at your own risk |
| `EXL3_ROCM_QUIET=1` | silence the "unmeasured GPU" warning |

## Known limits

- 3 bpw layers run at ~600 GB/s against a ~880 GB/s streaming ceiling; the loss is in the
  overlap of weight loads with the VALU decode, not in either alone. Deeper prefetch did not help.
- Prefill at long context is attention-bound (quadratic); the Triton tile was retuned for
  head_dim 256 on gfx11 (21 → 54 TFLOPS) but it is still 70% of the 192k prefill time.
- Batch > 1 rows cost 1.2×–2.1× a single row (no tensor cores); DFlash-style drafters that verify
  8–16 rows have not been tried.
- Greedy outputs are repeatable on one GPU but differ between GPU models after a few hundred
  tokens (different autotuned grids → different fp32 summation order).
