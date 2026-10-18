"""TTS Server: OpenAI compatible /v1/audio/speech minimal implementation"""
import io
import logging
import os
import threading
import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
#from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

MODEL_NAME = os.environ["TTS_MODEL"]
SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "Japanese")
INSTRUCT = "自然で落ち着いた話し方で"

MAX_SEQ_LEN = int(os.environ.get("TTS_MAX_SEQ_LEN", "2048"))
MAX_NEW = int(os.environ.get("TTS_MAX_NEW_TOKENS", "512"))   # about 40 s (12.5 frame per sec) to prevent from exhausting
CHUNK = int(os.environ.get("TTS_CHUNK_SIZE", "4"))

state: dict = {}
lock = threading.Lock()          # GPU one request once

def synth(text: str, speaker: str):
    with lock:
        wavs, sr = state["model"].generate_custom_voice(
            text=text, speaker=speaker, language=LANGUAGE, max_new_tokens=MAX_NEW, instruct=INSTRUCT)
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    return wav, sr


def start_stream(text: str, speaker: str):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    t0 = time.perf_counter()                                   # ★ 経過時間の計測

    def worker():
        n_chunks = 0                                           # ★ try の前で初期化
        got_lock = False                                       # ★ ロックを取れたか
        try:
            while not lock.acquire(timeout=0.1):               # 待機中の切断も検知する
                if cancel.is_set():
                    return                                     # ★ 取れずに終了（外側の finally は実行される）
            got_lock = True
            try:
                for chunk, sr, *_ in state["model"].generate_custom_voice_streaming(
                        text=text, speaker=speaker, language=LANGUAGE,
                        max_new_tokens=MAX_NEW, chunk_size=CHUNK, instruct=INSTRUCT):
                    if cancel.is_set():
                        break                                  # 生成を打ち切る
                    n_chunks += 1                              # ★ カウント
                    a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    pcm = (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
            finally:
                lock.release()                                 # 内側: ロックの解放
        finally:                                               # 外側: 終了の合図とログ
            log.info("stream worker done: chunks=%d locked=%s cancelled=%s elapsed=%.2fs",
                     n_chunks, got_lock, cancel.is_set(), time.perf_counter() - t0)   # ★
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def body():
        try:
            while (b := await q.get()) is not None:
                yield b
        finally:
            cancel.set()                                       # 切断・キャンセルでも必ず実行される
            log.info("stream body closed")                     # ★ HTTP 側が閉じた記録
    return body()

def pcm_stream(text: str, speaker: str):
    n, t0 = 0, time.perf_counter()
    try:
        with lock:
            for chunk, sr, *_ in state["model"].generate_custom_voice_streaming(
                    text=text, speaker=speaker, language=LANGUAGE,
                    max_new_tokens=MAX_NEW, chunk_size=CHUNK):
                n += 1
                a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                yield (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    finally:
        log.info("stream closed: chunks=%d elapsed=%.2fs", n, time.perf_counter() - t0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = FasterQwen3TTS.from_pretrained(MODEL_NAME, max_seq_len=MAX_SEQ_LEN)
    state["model"].warmup(prefill_len=100)
    #state["model"] = Qwen3TTSModel.from_pretrained(
    #    MODEL_NAME, device_map="cuda:0", dtype=torch.bfloat16,
    #    attn_implementation="sdpa",
    #)
    synth("こんにちは。", SPEAKER)                          # Warm up
    log.info("ready. gpu_mem_allocated=%.2fGB", torch.cuda.memory_allocated() / 1e9)
    yield


app = FastAPI(title="tts", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "wav"
    stream: bool = False

    
@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.stream:
        return StreamingResponse(start_stream(req.input, req.voice or SPEAKER),
                                 media_type="audio/pcm", headers={"X-Sample-Rate": "24000"})
    
    start = time.perf_counter()
    wav, sr = synth(req.input, req.voice or SPEAKER)
    elapsed = time.perf_counter() - start

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")                    # wav は numpy 配列を想定（tensor なら変換）
    seconds = len(wav) / sr
    log.info("synth %.2fs audio=%.2fs rtf=%.2f chars=%d", elapsed, seconds, elapsed / seconds, len(req.input))
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Synth-Seconds": f"{elapsed:.3f}", "X-Audio-Seconds": f"{seconds:.3f}"})
