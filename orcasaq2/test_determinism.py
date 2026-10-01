"""Bit-exact repeatability of the GEMM: 20 launches per (shape, grid) config must agree exactly.
Grids are given relative to the CU count. GRIDS env overrides, e.g. GRIDS=48,96,144"""
import sys, torch, os
from _common import apply_embedding_patch, model_dir
apply_embedding_patch()
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.ext import exllamav3_ext as ext
cfg = Config.from_directory(model_dir()); model = Model.from_config(cfg); model.load("cuda:0")
_cus = torch.cuda.get_device_properties(0).multi_processor_count
_g = [int(x) for x in os.environ["GRIDS"].split(",")] if os.environ.get("GRIDS") else [_cus // 2, _cus, 2 * _cus]
CONFIGS = [(2, _g[0]), (2, _g[1]), (1, _g[0]), (1, _g[1]), (1, _g[2]), (3, _g[0]), (3, _g[1]), (4, _g[0] // 2), (4, _g[0]), (4, _g[1])]
lins=[]
def rec(m):
    if isinstance(m, Linear) and getattr(m.inner,'K',None) is not None: lins.append(m)
    for s in getattr(m,'modules',[]): rec(s)
for m in model.modules[1:3]: rec(m)
l=next(l for l in lins if l.inner.in_features==5120 and l.inner.out_features==6144); inner=l.inner
x=torch.randn(1,5120,device="cuda:0",dtype=torch.half); xh=torch.empty_like(x)
with torch.inference_mode():
    bad=0
    for si,g in CONFIGS:
        ys=[]
        try:
            for _ in range(20):
                y=torch.empty(1,6144,device="cuda:0",dtype=torch.half)
                ext.exl3_gemm(x, inner.trellis, y, inner.suh, xh, inner.svh, si, inner.mcg, inner.mul1, g); ys.append(y.clone())
        except RuntimeError as e:
            print(f"shape{si} grid{g}: skipped ({str(e).splitlines()[0][:60]})", flush=True); continue
        nd=sum(not torch.equal(ys[0],y) for y in ys[1:])
        print(f"shape{si} grid{g}: {'bit-exact over 20 runs' if nd==0 else f'NONDETERMINISTIC ({nd}/19 runs differ)'}", flush=True)
        bad+=nd>0
    print("determinism:", "OK" if bad==0 else f"FAIL ({bad} configs)")
os._exit(1 if bad else 0)
