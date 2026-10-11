"""TTS Server: OpenAI compatible /v1/audio/speech minimal implementation"""
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

MODEL_NAME = os.environ["TTS_MODEL"]
SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "Japanese")
INSTRUCT = "自然で落ち着いた話し方で"

state: dict = {}
lock = threading.Lock()          # GPU one request once


def synth(text: str, speaker: str):
    with lock:
        wavs, sr = state["model"].generate_custom_voice(
            text=text, language=LANGUAGE, speaker=speaker, instruct=INSTRUCT
        )
    return wavs[0], sr


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = Qwen3TTSModel.from_pretrained(
        MODEL_NAME, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    synth("こんにちは。", SPEAKER)                          # Warm up
    log.info("ready. gpu_mem_allocated=%.2fGB", torch.cuda.memory_allocated() / 1e9)
    yield


app = FastAPI(title="tts", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "wav"


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading"}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    start = time.perf_counter()
    wav, sr = synth(req.input, req.voice or SPEAKER)
    elapsed = time.perf_counter() - start

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")                    # wav は numpy 配列を想定（tensor なら変換）
    seconds = len(wav) / sr
    log.info("synth %.2fs audio=%.2fs rtf=%.2f chars=%d", elapsed, seconds, elapsed / seconds, len(req.input))
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Synth-Seconds": f"{elapsed:.3f}", "X-Audio-Seconds": f"{seconds:.3f}"})
