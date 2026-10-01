"""Prefill / decode speed vs context length (Q8 KV cache, chunk 2048, greedy), optional MTP.
Usage: MTP=1 NDRAFT=3 CTX=8192,32768,196608 python bench_ctx.py   (defaults: MTP=0, CTX=8192,32768)
Needs a long text as prompt filler: README.md of this repo is used."""
import sys, time, torch, os, gc
from _common import apply_embedding_patch, load_model
apply_embedding_patch()
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
MTP = os.environ.get("MTP", "0") == "1"
ctxs = [int(c) for c in os.environ.get("CTX", "8192,32768").split(",")]
NEW = int(os.environ.get("NEW", 128))
CACHE = (max(ctxs) + 4096 + 255) // 256 * 256
draft_tokens = int(os.environ.get("NDRAFT", 3)) if MTP else 0
model, cache, tok, draft, dcache = load_model(CACHE, draft=MTP, max_history=draft_tokens)
print(f"MTP={MTP} draft_tokens={draft_tokens} cache={CACHE} tokens Q8", flush=True)
gen = Generator(model=model, cache=cache, draft_model=draft, draft_cache=dcache, tokenizer=tok, max_batch_size=1,
                max_chunk_size=2048, num_draft_tokens=draft_tokens)
free, total = torch.cuda.mem_get_info(0); print(f"VRAM after load: used {(total-free)/2**30:.1f} GiB of {total/2**30:.1f}", flush=True)
base = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")).read()
def make_prompt(n_tokens):
    text = base
    while len(tok.encode(text)[0]) < n_tokens + 64: text = text + "\n\n" + base
    ids = tok.encode(text)[:, :n_tokens - 16]
    tail = tok.encode("\n\n<|im_start|>user\nSummarize the document above in detail.<|im_end|>\n<|im_start|>assistant\n", encode_special_tokens=True)
    return torch.cat([ids, tail], dim=1)
res = []
for ctx in ctxs:
    ids = make_prompt(ctx); n_in = ids.shape[1]
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    job = Job(input_ids=ids, max_new_tokens=NEW, sampler=ArgmaxSampler())
    gen.enqueue(job); n = 0; t0 = time.time(); tfirst = None; out = ""; accepted = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if tfirst is None: tfirst = time.time()
                n += 1; out += r.get("text", "")
                if r.get("eos"): accepted = r.get("accepted_draft_tokens")
    torch.cuda.synchronize(); tend = time.time()
    prefill = tfirst - t0; dec = (tend - tfirst) / max(n, 1)
    free, _ = torch.cuda.mem_get_info(0)
    line = f"ctx {n_in:7d}: prefill {prefill:6.2f} s ({n_in/prefill:7.0f} tok/s) | decode {1/dec:6.2f} tok/s ({dec*1000:5.1f} ms/tok, {n} tok) | VRAM used {(total-free)/2**30:.1f} GiB"
    if accepted is not None: line += f" | accepted draft {accepted}"
    print(line, flush=True); res.append(line)
    print("   ", repr(out[:120]), flush=True)
print("\n".join(res)); os._exit(0)
