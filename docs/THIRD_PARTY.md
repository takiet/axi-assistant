# Third-Party Software and Models

このリポジトリ（VMS Voice Agent）が利用する第三者のソフトウェアとモデルの一覧です。
**このリポジトリには、第三者のソースコード、モデルの重み、ビルド済みのイメージは含まれていません。**
依存は `Dockerfile` や `pyproject.toml` の指定で、ビルド時と実行時に取得・参照します。

- 最終更新: 2026-10-18
- 確認方法: 各パッケージのメタデータ、モデルカード、公式 README
- 凡例: **確認済み** = 実物（メタデータ、モデルカード、README）で確認 / **要確認** = 未確認。記載の値は一般的な認識で、確認後に更新すること

---

## 1. TTS（音声合成）サーバ（`docker/tts-faster/Dockerfile`）

| 名前 | 版 | ライセンス | 状態 | 備考 |
|---|---|---|---|---|
| faster-qwen3-tts | 0.5.4 | MIT | 確認済み | CUDA Graph による Qwen3-TTS の高速化 |
| qwen-tts-hf | 0.1.1.post1 | Apache-2.0 | 確認済み | 非公式の互換ビルド（Transformers 5 対応）。`qwen_tts` を提供するため、上流の `qwen-tts` とは併用しない |
| transformers | 5.15.1 | Apache-2.0 | 確認済み | 5.16 以降は `MimiConfig` から `rope_theta` が消え、`qwen-tts-hf` が動作しないため 5.15.1 に固定 |
| huggingface-hub | 1.33.0 | Apache-2.0 | 確認済み | |
| torch | 2.14.0a0（NVIDIA ビルド） | Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT ほか | 確認済み | NVIDIA PyTorch コンテナに同梱 |
| torchaudio | 2.11.0（ソースからビルド） | BSD-2-Clause 系 | 要確認 | `pytorch/audio` の `v2.11.0` タグをビルド時に取得 |
| tokenizers | 0.22.2 | Apache-2.0 | 要確認 | transformers の依存として解決 |
| gradio / gradio_client | 6.30.0 / 2.7.2 | Apache-2.0 | 要確認 | `qwen-tts-hf` の依存宣言により導入。デモ UI 用で、本システムでは使用しない |
| fastapi | 0.143.0 | MIT | 要確認 | API サーバ |
| starlette | 1.7.0 | BSD-3-Clause | 要確認 | |
| uvicorn | 0.54.0 | BSD-3-Clause | 要確認 | |
| accelerate、librosa、soundfile、onnxruntime、sox、einops ほか | `docs/tts-faster-freeze.txt` を参照 | — | 要確認 | 版は freeze を正とする |

### モデルの重み

| 名前 | ライセンス | 状態 | 配置 |
|---|---|---|---|
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | Apache-2.0 | 確認済み（モデルカードの `license:`） | ホストの `/opt/models/tts/`（リポジトリ外） |

---

## 2. STT（音声認識）サーバ（`compose.yaml` の `vllm-stt`）

| 名前 | 版 | ライセンス | 状態 | 備考 |
|---|---|---|---|---|
| NVIDIA vLLM コンテナ（`nvcr.io/nvidia/vllm`） | 26.08-py3（vLLM 0.27.1） | NVIDIA 独自の許諾 ＋ 同梱 OSS | 要確認 | イメージの再配布は別途確認 |
| Qwen/Qwen3-ASR-1.7B（重み） | **未確認** | 要確認 | モデルカードの `license:` を確認 | ホストの `/opt/models/hf/`（リポジトリ外） |

---

## 3. アプリケーション層（`docker/crewai/Dockerfile`、`pyproject.toml`）

| 名前 | 版 | ライセンス | 状態 |
|---|---|---|---|
| crewai | 1.15.23 | MIT | 要確認 |
| openai（Python SDK） | `uv.lock` を参照 | Apache-2.0 | 要確認 |
| fastapi / uvicorn | `uv.lock` を参照 | MIT / BSD-3-Clause | 要確認 |
| pytest（開発用） | `uv.lock` を参照 | MIT | 要確認 |

---

## 4. ベースイメージ

| イメージ | ライセンス | 備考 |
|---|---|---|
| `nvcr.io/nvidia/pytorch:26.09-py3` | NVIDIA Software License Agreement ＋ Product-Specific Terms for NVIDIA AI Products | 起動時のバナーに記載。**イメージの再配布の可否は、許諾を確認してから**行う |
| `nvcr.io/nvidia/vllm:26.08-py3` | 同上 | 同上 |
| `python:3.12-slim` | PSF License ＋ Debian の各パッケージのライセンス | |

---

## 5. 既知の事項

- `pip check` が、NVIDIA ベースイメージ由来の 4 行（`triton-kernels`、`cupti-python`、`flash-attn-4`、`quack-kernels`）を報告する。ベースイメージ単体でも同じで、このプロジェクトで増やしたものではない。
- `gradio` は、依存の宣言どおりに導入される。デモ UI 用で、使用しない。

---

## 6. 運用ルール

| 場面 | 対応 |
|---|---|
| 依存の版を更新した | この表と `docs/tts-faster-freeze.txt` を更新する |
| モデルを追加・変更した | **そのモデルの `license:` をモデルカードで確認**し、この表に追記する |
| イメージ、重み、ソースを**公開・配布**する | 各ライセンス文と NOTICE の同梱、改変の明示、NVIDIA 許諾の再配布条件を確認する |
| 第三者のコードを**取り込んで改変**した | 元のライセンス表記と著作権表示を保持し、改変した旨を記録する |
| 業務・商用で使う | 社内規定と、各ライセンスの条件に照らして最終確認する |

---

## 7. 実物の確認コマンド

```bash
# パッケージのライセンス（メタデータ）
docker run --rm vms/tts-faster:dev python3 -c "
import importlib.metadata as m
names = ['faster-qwen3-tts','qwen-tts-hf','transformers','huggingface-hub','tokenizers','torch',
         'torchaudio','gradio','fastapi','starlette','uvicorn','accelerate','librosa',
         'soundfile','onnxruntime','sox','einops']
for n in names:
    try:
        md = m.metadata(n)
        lic = md.get('License-Expression') or md.get('License') or \
              [c for c in md.get_all('Classifier', []) if 'License' in c]
        print(f'{n:22s} {m.version(n):28s} {lic}')
    except m.PackageNotFoundError:
        print(f'{n:22s} (not installed)')"

# モデルのライセンス（ローカルのモデルカード）
grep -i -m3 "license" /opt/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice/README.md
```

> メタデータの `License` 欄が空や長文のときは、各プロジェクトの `LICENSE` ファイルや公式ページで確認する。
