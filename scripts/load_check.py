"""モデルの読み込みと CUDA graph の取得ができるかの確認（速度は測らない）"""
import os
import time

import transformers
from faster_qwen3_tts import FasterQwen3TTS

MODEL = os.environ.get("TTS_MODEL", "/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice")
print("transformers", transformers.__version__)

t = time.perf_counter()
model = FasterQwen3TTS.from_pretrained(MODEL, max_seq_len=512)
print(f"load OK {time.perf_counter() - t:.1f}s")

t = time.perf_counter()
model.warmup(prefill_len=100)
print(f"warmup OK {time.perf_counter() - t:.1f}s")
