"""faster-qwen3-tts のベンチ（CustomVoice / 日本語 / 非ストリーミング + ストリーミング）"""
import inspect
import os
import statistics
import time

import torch
from faster_qwen3_tts import FasterQwen3TTS

MODEL = os.environ.get("TTS_MODEL", "/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice")
SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "Japanese")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
RUNS = int(os.environ.get("BENCH_RUNS", "5"))
INSTRUCT = os.environ.get("BENCH_INSTRUCT") or None      # 例: "少し早口で、はっきりと話してください。"
MAX_NEW = 256                                            # 約 20 秒分（12.5 フレーム/秒）
FPS = 12.5                                               # 12Hz モデルのフレームレート

TEXTS = [
    "確認します。",
    "入口カメラに人が一人います。",
    "搬入口カメラと荷捌き場のカメラを表示します。",
]


def sync():
    torch.cuda.synchronize()


def n_samples(wav) -> int:
    return int(wav.shape[-1]) if hasattr(wav, "shape") else len(wav)


def synth(text: str):
    sync()
    t = time.perf_counter()
    out = model.generate_custom_voice(
        text=text, speaker=SPEAKER, language=LANGUAGE,
        instruct=INSTRUCT, max_new_tokens=MAX_NEW,
    )
    sync()
    elapsed = time.perf_counter() - t
    wavs, sr = out
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    return elapsed, n_samples(wav) / sr


print(f"model={MODEL}  max_seq_len={MAX_SEQ_LEN}  instruct={INSTRUCT!r}")
t0 = time.perf_counter()
model = FasterQwen3TTS.from_pretrained(MODEL, max_seq_len=MAX_SEQ_LEN)   # ローカルのパスを渡す
print(f"load: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
model.warmup(prefill_len=100)                                            # CUDA graph の取得
sync()
print(f"warmup: {time.perf_counter() - t0:.1f}s   gpu_mem_allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB")

print("\n=== 非ストリーミング ===  （基準: 素の実装は約 100ms/フレーム、RTF 1.1〜1.3）")
for text in TEXTS:
    for _ in range(2):
        synth(text)                                                      # ウォームアップ
    rows = [synth(text) for _ in range(RUNS)]
    el = statistics.median(r[0] for r in rows)
    sec = statistics.median(r[1] for r in rows)
    print(f"{text[:14]:16s} synth={el:.2f}s audio={sec:.2f}s  "
          f"ms/frame={el / (sec * FPS) * 1000:.0f}  rtf(ours)={el / sec:.2f}  rtf(readme)={sec / el:.2f}")

print("\n=== ストリーミング ===")
sig = inspect.signature(model.generate_custom_voice_streaming)
print("params:", list(sig.parameters))
if "chunk_size" in sig.parameters:
    text = TEXTS[2]
    for cs in (2, 4, 8):
        for trial in range(2):                                           # 2 回目の値を見る
            sync()
            t = time.perf_counter()
            first, total = None, 0.0
            for chunk, sr, *_ in model.generate_custom_voice_streaming(
                text=text, speaker=SPEAKER, language=LANGUAGE,
                instruct=INSTRUCT, max_new_tokens=MAX_NEW, chunk_size=cs,
            ):
                if first is None:
                    sync()
                    first = time.perf_counter() - t
                total += n_samples(chunk) / sr
            sync()
            elapsed = time.perf_counter() - t
        print(f"chunk_size={cs}: TTFA={first * 1000:.0f}ms  total={elapsed:.2f}s  audio={total:.2f}s  rtf(ours)={elapsed / total:.2f}")
else:
    print("chunk_size 引数が見つかりません（署名を確認して、貼ってください）")
