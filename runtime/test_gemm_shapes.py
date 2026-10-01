"""Correctness of every distinct decoder GEMM shape against a float reference, M rows (M=1 default).
Prints before each launch so a hung kernel can be attributed to a shape; run under `timeout`,
preferably with a -DEXL3_ROCM_BOUNDS build (see check_device.sh)."""
import sys, torch, os
from _common import apply_embedding_patch, model_dir
apply_embedding_patch()
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.ext import exllamav3_ext as ext
cfg = Config.from_directory(model_dir()); model = Model.from_config(cfg); model.load("cuda:0")
def all_linears(mod, out):
    if isinstance(mod, Linear): out.append(mod)
    for sm in getattr(mod, 'modules', []): all_linears(sm, out)
lins = []
for m in model.modules[1:65]: all_linears(m, lins)
lins = [l for l in lins if getattr(l.inner, 'K', None) is not None]
only = os.environ.get("ONLY")   # e.g. "3.5"
seen = {}
for l in lins:
    k = (l.inner.K, l.inner.in_features, l.inner.out_features)
    if k not in seen: seen[k] = l
M = int(os.environ.get("M", 1))
with torch.inference_mode():
    for k, l in sorted(seen.items()):
        if only and str(k[0]) != only: continue
        inner = l.inner
        x = torch.randn(M, inner.in_features, device="cuda:0", dtype=torch.half)
        y = torch.empty(M, inner.out_features, device="cuda:0", dtype=torch.half); xh = torch.empty_like(x)
        print(f"K={k[0]:<4} {k[1]:5d}x{k[2]:6d} ...", end=" ", flush=True)
        ext.exl3_gemm(x, inner.trellis, y, inner.suh, xh, inner.svh, -1, inner.mcg, inner.mul1, 0)
        torch.cuda.synchronize()
        W = inner.get_weight_tensor(); yref = x.float() @ W.float()
        err = ((y.float() - yref).norm() / yref.norm()).item()
        print(f"ok  rel_err={err:.5f}{'  BAD' if err > 0.002 else ''}", flush=True)
print("all done", flush=True)
sys.stdout.flush(); os._exit(0)   # _exit skips the (slow) CUDA teardown; flush first so piped output is not lost
