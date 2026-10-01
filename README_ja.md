# OrcaSAQ2-27B on AMD Radeon — ROCm ランタイム

[English](README.md) | 日本語

[OrcaSAQ2-27B](https://huggingface.co/Continuum-AI-Corp/OrcaSAQ2-27B)(Qwen3.5-27B 級のモデルを
EXL3 トレリス形式で約 3.2 bit/weight に量子化したもの)を、**RDNA3 / RDNA4** の Radeon 1 枚で
**38–52 tokens/s** で動かし、TabbyAPI 経由で OpenAI 互換 API として提供するためのランタイムです。

[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) のフォークで、
[Calandracas606 氏の ROCm 移植](https://github.com/Calandracas606/exllamav3) をベースにしています。
EXL3 GEMM カーネルを RDNA のベクタ ALU 向けに書き直し(このデータ配置に使えるテンサーコア経路は
存在しません)、GatedDeltaNet と norm/residual の融合でカーネル起動回数を削減し、MTP 投機デコードが
効くようにしました。それ以外は upstream exllamav3 v1.5.1 のままです。

| コンテキスト | RX 7900 XTX (24 GB) | MTP あり | Radeon AI PRO R9700 (32 GB) | MTP あり |
|---:|---:|---:|---:|---:|
| 8k   | 38.0 tok/s | **51.9** | 36.2 | **49.8** |
| 32k  | 30.8 | **45.3** | 29.9 | **49.8** |
| 192k | 13.5 | **21.3** | 14.6 | **22.1** |

プリフィル: 8–32k で 1100–1800 tok/s、192k で約 700–950 tok/s。出力はビット単位で再現可能です。
詳細・測定方法・カーネルの仕組み: [runtime/PERFORMANCE.md](runtime/PERFORMANCE.md)(英語)。

## 対応 GPU

| 状態 | GPU |
|---|---|
| **実測済み** | RX 7900 XTX (gfx1100) · Radeon AI PRO R9700 (gfx1201) |
| 動くはずだが未測定 | その他の RDNA3 / RDNA4: 7900 XT / GRE、7800 XT、7700、7600、W7800/W7900、9070 / 9060。gfx ターゲットを `PYTORCH_ROCM_ARCH` に追加し、後述の `check_device.sh` を実行してください。結果報告歓迎 |
| 非対応 | Instinct / CDNA(wave64、MFMA カーネルが必要)、RDNA2 以前。拡張モジュールはロードを拒否します |

必要なもの: Linux、8k コンテキストで約 15 GB の空き VRAM(24 GB カードでロード後 17 GB 使用)、
Python ≥ 3.10、[uv](https://docs.astral.sh/uv/)、HIP ビルド用のホスト C++ ツールチェーン
(libstdc++ ヘッダ付きの `g++`: `sudo apt install g++` / `dnf install gcc-c++`)。システムへの ROCm
インストールは不要です — `rocm` extra が PyTorch、ROCm SDK(clang/hipcc)、デバイスライブラリを
wheel として取得します。カーネルドライバ(`amdgpu`、`/dev/kfd`)が存在し、ユーザーが `render`/`video`
グループに入っている必要があります。

## インストール

```sh
git clone https://github.com/Argos1111/orcasaq2-rocm
cd orcasaq2-rocm
export PYTORCH_ROCM_ARCH="gfx1100;gfx1201"        # 手持ちのターゲットのみ。1 つ増えるごとにビルド +50 秒程度
uv venv && uv sync --extra rocm --no-install-project
uv sync --extra rocm --no-build-isolation         # HIP 拡張をビルド(約 2 分)

git clone https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel   # int8 埋め込みテーブル用のローダパッチ
hf download Continuum-AI-Corp/OrcaSAQ2-27B --local-dir models/OrcaSAQ2-27B
```

> このチェックポイントは埋め込みテーブルを int8 で保存しています(1.3 GB 節約)。exllamav3 が
> それを読むには `OrcaSAQ2-kernel` の小さなローダパッチが必要です。以下のスクリプトは
> `ORCASAQ2_KERNEL`(既定 `./OrcaSAQ2-kernel`)でパッチを、`ORCASAQ2_MODEL`(既定
> `./models/OrcaSAQ2-27B`)でモデルを見つけます。

### 最初の実行

```sh
HIP_VISIBLE_DEVICES=0 uv run --extra rocm runtime/bench_decode.py          # 約 3 分。4 ラウンドの tok/s を表示
HIP_VISIBLE_DEVICES=0 MTP=1 uv run --extra rocm runtime/bench_decode.py    # 投機デコードあり
```

> `uv run` には必ず `--extra rocm` を付けてください(または `.venv/bin/python` を直接呼ぶ)。
> 素の `uv run` は環境をプロジェクト既定の CUDA 版 torch に再同期してしまい、拡張モジュールが
> `ImportError: libamdhip64.so.7` で失敗します。

実測済みリストにない GPU では、代わりにファーストコンタクト用チェックを実行してください。
デバッグトラップ付きでビルドするので、カーネルの前提が外れていてもカードがハングせずプロセスが
終了します。続けて全 GEMM 形状の検証、ビット再現性、速度を確認します(約 10 分):

```sh
HIP_VISIBLE_DEVICES=0 ./runtime/check_device.sh
```

## TabbyAPI でサーブする

```sh
git clone https://github.com/theroyallab/tabbyAPI
uv pip install --python .venv/bin/python "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich uvicorn \
    jinja2 loguru sse-starlette packaging tokenizers numpy aiofiles aiohttp async_lru huggingface_hub \
    psutil httptools pillow requests uvloop                  # TabbyAPI の依存をこの venv に入れる(TabbyAPI 自身の venv は使わない)
cp runtime/tabbyapi-config.yml tabbyAPI/config.yml       # Q8 キャッシュ、64k コンテキスト、MTP draft 3
cp runtime/tabbyapi_main.py   tabbyAPI/
ln -s "$PWD/models/OrcaSAQ2-27B" tabbyAPI/models/
cd tabbyAPI && HIP_VISIBLE_DEVICES=0 ORCASAQ2_KERNEL=../OrcaSAQ2-kernel ../.venv/bin/python tabbyapi_main.py
```

API は `http://127.0.0.1:5000/v1` で待ち受けます(OpenAI 互換、モデル ID は `OrcaSAQ2-27B`)。
`tabbyapi_main.py` は TabbyAPI の `main.py` の代わりに使います: OrcaSAQ2 の埋め込みパッチを適用し、
TabbyAPI の exllamav3 バックエンドが ROCm 上で起動できるようにし(upstream TabbyAPI は、upstream
exllamav3 が ROCm 非対応であるため AMD GPU を拒否します)、その後 `main.py` に処理を渡します。
TabbyAPI の `start.py`/`start.sh` は実行しないでください — CUDA 版 torch 入りの独自 venv を作って
しまいます。VRAM 配分のつまみ(`gpu_split`、`cache_size`)は 24 GB / 32 GB カード向けに config の
コメントで説明しています。

## `runtime/` の中身

| ファイル | 用途 |
|---|---|
| `bench_decode.py` | デコード tok/s。`MTP=1 NDRAFT=3` で投機デコード |
| `bench_ctx.py` | コンテキスト長ごとのプリフィル + デコード。例: `CTX=8192,32768,196608 MTP=1` |
| `check_device.sh` | 未測定 GPU 向けファーストコンタクトチェック |
| `test_gemm_shapes.py`, `test_determinism.py` | 上記チェックが実行する正当性・再現性テスト |
| `tabbyapi-config.yml`, `tabbyapi_main.py` | TabbyAPI 設定 |
| `PERFORMANCE.md` | 測定結果、カーネル設計、環境変数、既知の制限 |

## スコープ

このリポジトリは OrcaSAQ2-27B を Radeon で速く動かすためのものです。他の EXL3 モデルも同じコードで
ロードできますが(upstream exllamav3 の挙動)、ここでは他に何も測定しておらず、GatedDeltaNet の
融合は Qwen3.5 系アーキテクチャにのみ適用されます。ROCm での一般的な exllamav3 利用は
Calandracas606 氏のフォークを、CUDA では upstream をご利用ください。

upstream のドキュメント(CUDA でのインストール、変換、対応アーキテクチャ):
[doc/README.upstream.md](doc/README.upstream.md)。

## ライセンス

MIT(upstream exllamav3 と同じ、© Turboderp)。OrcaSAQ2 のモデル重みと OrcaSAQ2-kernel のパッチは
それぞれのライセンスに従います(各リポジトリを参照)。
