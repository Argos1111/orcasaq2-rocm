"""TabbyAPI entry point for OrcaSAQ2 checkpoints.

OrcaSAQ2 packs the embedding table as int8 rows + per-row scales (embed_tokens.qweight /
.scales) instead of the bf16 `embed_tokens.weight` exllamav3 expects. The OrcaSAQ2-kernel repo
ships a loader patch for exllamav3's Embedding module; it must be applied before the first
Config.from_directory, so this wrapper installs it and then hands over to TabbyAPI's main.py.

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
import runpy  # noqa: E402
runpy.run_path(os.path.join(HERE, "main.py"), run_name="__main__")
