"""Shared prelude for the OrcaSAQ2 scripts in this directory.

Environment:
  ORCASAQ2_MODEL   path to the OrcaSAQ2-27B checkpoint directory (default: ./models/OrcaSAQ2-27B)
  ORCASAQ2_KERNEL  path to a checkout of https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel
                   (its exllamav3 int8-embedding loader patch is required; default: ./OrcaSAQ2-kernel)
  HIP_VISIBLE_DEVICES  which GPU to use (one at a time)
"""
import os, sys

def model_dir():
    p = os.environ.get("ORCASAQ2_MODEL", "models/OrcaSAQ2-27B")
    if not os.path.isfile(os.path.join(p, "config.json")):
        sys.exit(f"OrcaSAQ2 checkpoint not found at {p!r}; set ORCASAQ2_MODEL")
    return p

def apply_embedding_patch():
    k = os.environ.get("ORCASAQ2_KERNEL", "OrcaSAQ2-kernel")
    if os.path.isdir(os.path.join(k, "orcasaq2")):
        sys.path.insert(0, k)
    try:
        from orcasaq2.patches import int8_embedding
    except ImportError:
        sys.exit("OrcaSAQ2-kernel not found: clone https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel "
                 "and set ORCASAQ2_KERNEL (its int8 embedding patch is needed to load the checkpoint)")
    int8_embedding.apply()

def load_model(cache_tokens, cache_bits=8, draft=False, max_history=0):
    """Returns (model, cache, tokenizer, draft_model, draft_cache). Q8 KV cache like the TabbyAPI config."""
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    cfg = Config.from_directory(model_dir())
    model = Model.from_config(cfg)
    kw = dict(layer_type=CacheLayer_quant, k_bits=cache_bits, v_bits=cache_bits, max_batch_size=1, max_history=max_history)
    cache = Cache(model, max_num_tokens=cache_tokens, **kw)
    model.load("cuda:0")
    dm = dc = None
    if draft:
        dm = Model.from_config(cfg, component="mtp")
        dc = Cache(dm, max_num_tokens=cache_tokens, **kw)
        dm.load("cuda:0")
    return model, cache, Tokenizer.from_config(cfg), dm, dc
