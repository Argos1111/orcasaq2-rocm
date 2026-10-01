"""TabbyAPI entry point for OrcaSAQ2 checkpoints.

OrcaSAQ2 packs the embedding table as int8 rows + per-row scales (embed_tokens.qweight /
.scales) instead of the bf16 `embed_tokens.weight` exllamav3 expects. The OrcaSAQ2-kernel repo
ships a loader patch for exllamav3's Embedding module; it has to be applied before the first
Config.from_directory, so this wrapper installs it and then hands over to TabbyAPI's main.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "orcasaq2-kernel"))

from orcasaq2.patches import int8_embedding  # noqa: E402

int8_embedding.apply()

if os.environ.get("ORCA_PROBE"):
    exec(open(os.environ["ORCA_PROBE"]).read())

sys.path.insert(0, HERE)
os.chdir(HERE)

import runpy  # noqa: E402

runpy.run_path(os.path.join(HERE, "main.py"), run_name="__main__")
