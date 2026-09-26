#pragma once

#if defined(USE_ROCM)
// Measured number of blocks per CU that are guaranteed to be co-resident for `kernel` launched
// with `block_dim` threads and `dyn_smem` bytes of dynamic LDS (see rocm_coresidency.cu). Use
// this - not cudaOccupancyMaxActiveBlocksPerMultiprocessor - to bound the grid of a plain-launched
// kernel whose blocks wait on each other.
int rocm_coresident_blocks_per_cu(const void* kernel, int block_dim, int dyn_smem);
#endif
