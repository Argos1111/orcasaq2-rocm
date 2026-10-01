"""TabbyAPI entry point for OrcaSAQ2 checkpoints.

Two things upstream TabbyAPI does not do, applied before handing over to its main.py:

1. OrcaSAQ2 packs the embedding table as int8 rows + per-row scales (embed_tokens.qweight /
   .scales) instead of the bf16 `embed_tokens.weight` exllamav3 expects. The OrcaSAQ2-kernel repo
   ships a loader patch for exllamav3's Embedding module (must be applied before the first
   Config.from_directory).
2. TabbyAPI's exllamav3 backend refuses to start on ROCm ("AMD GPUs are not supported" —
   true for upstream exllamav3, not for this fork). `common.hardware.hardware_supports_exllamav3`
   is replaced so HIP devices pass; the CUDA compute-capability floor is kept for CUDA devices.

Copy this file into your TabbyAPI checkout next to main.py and run it instead of main.py:
    ORCASAQ2_KERNEL=/path/to/OrcaSAQ2-kernel python tabbyapi_main.py --config config.yml
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# OrcaSAQ2-kernel checkout: $ORCASAQ2_KERNEL, else ../OrcaSAQ2-kernel relative to the TabbyAPI dir
kernel_dir = os.path.abspath(os.environ.get("ORCASAQ2_KERNEL", os.path.join(os.path.dirname(HERE), "OrcaSAQ2-kernel")))
sys.path.insert(0, kernel_dir)
try:
    from orcasaq2.patches import int8_embedding  # noqa: E402
except ImportError:
    sys.exit(f"OrcaSAQ2-kernel not found at {kernel_dir}: clone "
             "https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel and set ORCASAQ2_KERNEL")
int8_embedding.apply()

sys.path.insert(0, HERE)
os.chdir(HERE)

import torch  # noqa: E402
import common.hardware as _hw  # noqa: E402

_upstream_check = _hw.hardware_supports_exllamav3


def _hardware_supports_exllamav3(gpu_device_list):
    if torch.version.hip:
        return True
    return _upstream_check(gpu_device_list)


_hw.hardware_supports_exllamav3 = _hardware_supports_exllamav3  # backends.exllamav3.model imports it from here later

import runpy  # noqa: E402
runpy.run_path(os.path.join(HERE, "main.py"), run_name="__main__")
