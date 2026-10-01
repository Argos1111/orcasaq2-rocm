"""Decode benchmark: 4 rounds of 256 greedy tokens from a short prompt; round 1 includes the
autotuner warm-up, rounds 2-4 are steady state. Optional MTP: MTP=1 [NDRAFT=3].
Usage: ORCASAQ2_MODEL=... ORCASAQ2_KERNEL=... HIP_VISIBLE_DEVICES=0 python bench_decode.py"""
import sys, time, torch, os
from _common import apply_embedding_patch, load_model
apply_embedding_patch()
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
MTP = os.environ.get("MTP", "0") == "1"; NDRAFT = int(os.environ.get("NDRAFT", 3))
model, cache, tok, dm, dc = load_model(8192, draft=MTP, max_history=NDRAFT if MTP else 0)
gen = Generator(model=model, cache=cache, draft_model=dm, draft_cache=dc, tokenizer=tok, max_batch_size=1,
                max_chunk_size=2048, num_draft_tokens=NDRAFT if MTP else 0)
prompt = "<|im_start|>user\nWrite a long, detailed essay about the history of the Roman Empire.<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(prompt, encode_special_tokens=True)
res=[]
for rnd in range(4):
    job = Job(input_ids=ids, max_new_tokens=256, sampler=ArgmaxSampler())
    gen.enqueue(job); n=0; tfirst=None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage")=="streaming":
                if tfirst is None: tfirst=time.time()
                n+=1
    torch.cuda.synchronize(); dt=time.time()-tfirst; res.append(n/dt)
print("decode tok/s per round:", " ".join(f"{r:.2f}" for r in res))
os._exit(0)
