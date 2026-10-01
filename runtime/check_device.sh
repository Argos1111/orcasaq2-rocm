#!/usr/bin/env bash
# First-contact check for a GPU this fork has not been measured on. ~10 minutes.
#   ORCASAQ2_MODEL=models/OrcaSAQ2-27B ORCASAQ2_KERNEL=OrcaSAQ2-kernel HIP_VISIBLE_DEVICES=0 ./runtime/check_device.sh
# 1. debug build (-DEXL3_ROCM_BOUNDS: out-of-range tile reads and lock waits trap instead of hanging the GPU)
# 2. every decoder GEMM shape at M = 1, 3, 8 against a float reference
# 3. bit-exact repeatability across runs
# 4. release build + decode benchmark
# Everything runs under `timeout`: a hung kernel ends the process, not the machine.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"   # the project venv (uv sync --extra rocm); plain `uv run` would re-sync to CUDA torch
rebuild() {  # $1 = extra hipcc flags
  find exllamav3/exllamav3_ext -name "*_hip.*" -o -name "*.hip" | xargs -r rm -f
  HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:-} $1" uv sync --inexact --extra rocm --no-build-isolation --reinstall-package exllamav3 > /tmp/orcasaq2_build.log 2>&1 \
    || { grep -n "error:" /tmp/orcasaq2_build.log | head -20; return 1; }
  rm -f ~/.cache/exllamav3/autotune/coop_autotune_v1.bin
}
echo "== device: $($PY -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name, p.gcnArchName, p.multi_processor_count, 'CUs', round(p.total_memory/2**30,1), 'GiB')")"
echo "== 1/4 debug build"; rebuild -DEXL3_ROCM_BOUNDS || exit 1
for M in 1 3 8; do
  n=$(M=$M timeout 900 "$PY" -u runtime/test_gemm_shapes.py 2>&1 | grep -cE 'ok  rel_err=0\.000')
  echo "== 2/4 correctness M=$M: $n/16 shapes ok"; [ "$n" = 16 ] || { echo "FAIL (rerun: M=$M $PY runtime/test_gemm_shapes.py)"; exit 1; }
done
timeout 900 "$PY" -u runtime/test_determinism.py 2>&1 | grep -E "^shape|^determinism" | sed 's/^/== 3\/4 /' | tail -3
timeout 900 "$PY" -u runtime/test_determinism.py > /dev/null 2>&1 || { echo "FAIL (rerun: $PY runtime/test_determinism.py)"; exit 1; }
echo "== 4/4 release build + decode benchmark"; rebuild "" || exit 1
timeout 900 "$PY" -u runtime/bench_decode.py 2>&1 | tail -1
echo "== done. Please include the lines above when reporting results for an untested GPU."
