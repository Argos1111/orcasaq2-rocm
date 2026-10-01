# OrcaSAQ2-27B on AMD Radeon — ROCm runtime

Run [OrcaSAQ2-27B](https://huggingface.co/Continuum-AI-Corp/OrcaSAQ2-27B) (a Qwen3.5-27B-class model
quantised to ~3.2 bits/weight in the EXL3 trellis format) on a single **RDNA3 / RDNA4** Radeon at
**38–52 tokens/s**, with an OpenAI-compatible API through TabbyAPI.

This is a fork of [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), built on
[Calandracas606's ROCm port](https://github.com/Calandracas606/exllamav3). The EXL3 GEMM kernels
were rewritten for RDNA's vector ALUs (no tensor-core path exists for this data layout), launch
counts were cut by fusing the GatedDeltaNet and norm/residual steps, and MTP speculative decoding
was made to pay off. Everything else is upstream exllamav3 v1.5.1.

| context | RX 7900 XTX (24 GB) | with MTP | Radeon AI PRO R9700 (32 GB) | with MTP |
|---:|---:|---:|---:|---:|
| 8k   | 38.0 tok/s | **51.9** | 36.2 | **49.8** |
| 32k  | 30.8 | **45.3** | 29.9 | **49.8** |
| 192k | 13.5 | **21.3** | 14.6 | **22.1** |

Prefill: 1100–1800 tok/s at 8–32k, ~700–950 tok/s at 192k. Outputs are bit-exact repeatable.
Details, methodology and how the kernels work: [runtime/PERFORMANCE.md](runtime/PERFORMANCE.md).

## Supported GPUs

| status | GPUs |
|---|---|
| **measured** | RX 7900 XTX (gfx1100) · Radeon AI PRO R9700 (gfx1201) |
| expected to work, unmeasured | other RDNA3 / RDNA4: 7900 XT / GRE, 7800 XT, 7700, 7600, W7800/W7900, 9070 / 9060. Add the gfx target to `PYTORCH_ROCM_ARCH` and run `check_device.sh` (below); results welcome |
| not supported | Instinct / CDNA (wave64, needs MFMA kernels), RDNA2 and older. The extension refuses to load on them |

Requirements: Linux, ~15 GB free VRAM for 8k context (17 GB used after load on a 24 GB card),
Python ≥ 3.10, [uv](https://docs.astral.sh/uv/). No system ROCm install needed — the `rocm` extra
pulls PyTorch, the ROCm SDK and device libraries as wheels.

## Install

```sh
git clone https://github.com/Argos1111/orcasaq2-rocm
cd orcasaq2-rocm
export PYTORCH_ROCM_ARCH="gfx1100;gfx1201"        # the targets you own; each adds ~50 s to the build
uv venv && uv sync --extra rocm --no-install-project
uv sync --extra rocm --no-build-isolation         # builds the HIP extension (~2 min)

git clone https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel   # loader patch for the int8 embedding table
hf download Continuum-AI-Corp/OrcaSAQ2-27B --local-dir models/OrcaSAQ2-27B
```

> The checkpoint stores its embedding table as int8 (saving 1.3 GB); exllamav3 needs the small
> loader patch from `OrcaSAQ2-kernel` to read it. The scripts below find it via `ORCASAQ2_KERNEL`
> (default `./OrcaSAQ2-kernel`) and the model via `ORCASAQ2_MODEL` (default `./models/OrcaSAQ2-27B`).

### First run

```sh
HIP_VISIBLE_DEVICES=0 uv run --extra rocm runtime/bench_decode.py          # ~3 min; tok/s for 4 rounds
HIP_VISIBLE_DEVICES=0 MTP=1 uv run --extra rocm runtime/bench_decode.py    # with speculative decoding
```

> Always pass `--extra rocm` to `uv run` (or call `.venv/bin/python` directly): a bare `uv run`
> re-syncs the environment to the project's default CUDA torch and the extension then fails with
> `ImportError: libamdhip64.so.7`.

On a GPU that is not in the measured list, run the first-contact check instead. It builds with
debug traps so a wrong kernel assumption ends the process rather than hanging the card, then
checks every GEMM shape, bit-exact repeatability and speed (~10 min):

```sh
HIP_VISIBLE_DEVICES=0 ./runtime/check_device.sh
```

## Serve with TabbyAPI

```sh
git clone https://github.com/theroyallab/tabbyAPI
cp runtime/tabbyapi-config.yml tabbyAPI/config.yml       # Q8 cache, 64k context, MTP draft 3
cp runtime/tabbyapi_main.py   tabbyAPI/
ln -s "$PWD/models" tabbyAPI/models
cd tabbyAPI && HIP_VISIBLE_DEVICES=0 ORCASAQ2_KERNEL=../OrcaSAQ2-kernel ../.venv/bin/python tabbyapi_main.py
```

The API is then at `http://127.0.0.1:5000/v1`. `tabbyapi_main.py` applies the embedding patch and
hands over to TabbyAPI's `main.py`; use it in place of `main.py`. The config comments explain the
VRAM budget knobs (`gpu_split`, `cache_size`) for 24 GB vs 32 GB cards.

## What's in `runtime/`

| file | purpose |
|---|---|
| `bench_decode.py` | decode tok/s, optional `MTP=1 NDRAFT=3` |
| `bench_ctx.py` | prefill + decode vs context length, e.g. `CTX=8192,32768,196608 MTP=1` |
| `check_device.sh` | first-contact check for an unmeasured GPU |
| `test_gemm_shapes.py`, `test_determinism.py` | the correctness and repeatability tests the check runs |
| `tabbyapi-config.yml`, `tabbyapi_main.py` | TabbyAPI setup |
| `PERFORMANCE.md` | measurements, kernel design, environment variables, known limits |

## Scope

This repository exists to run OrcaSAQ2-27B fast on Radeon. Other EXL3 models load through the
same code (upstream exllamav3 behaviour), but nothing else has been measured here and the
GatedDeltaNet fusions only apply to Qwen3.5-style architectures. For general exllamav3 use on
ROCm see Calandracas606's fork; for CUDA use upstream.

Upstream documentation (installation on CUDA, conversion, supported architectures):
[doc/README.upstream.md](doc/README.upstream.md).

## License

MIT, as upstream exllamav3 (© Turboderp). OrcaSAQ2 model weights and the OrcaSAQ2-kernel patch
are under their own licenses (see their repositories).
