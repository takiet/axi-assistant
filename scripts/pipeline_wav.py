"""Phase 1-B: WAV → STT → LLM → TTS → WAV（区間ごとの時間を記録）"""
import os
import sys
import time
from pathlib import Path

from crewai import LLM
from openai import OpenAI

STT = OpenAI(base_url=os.environ.get("STT_BASE_URL", "http://vllm-stt:6180/v1"), api_key="dummy", timeout=120)
TTS = OpenAI(base_url=os.environ.get("TTS_BASE_URL", "http://tts:6280/v1"), api_key="dummy", timeout=120)
LLM_ = LLM(
    model=os.environ["ORCHESTRATOR_MODEL"],
    base_url=os.environ["ORCHESTRATOR_BASE_URL"],
    api_key=os.environ["ORCHESTRATOR_API_KEY"],
)
SYSTEM = Path("vms_agent/prompts/voice_direct.txt").read_text()   # 上のシステムプロンプト
VOICE = os.environ.get("TTS_SPEAKER", "Vivian")


def timed(label, fn, times):
    start = time.perf_counter()
    result = fn()
    times[label] = time.perf_counter() - start
    return result


def run(wav_path: str, out_path: str) -> dict:
    times: dict[str, float] = {}
    t0 = time.perf_counter()

    def stt():
        with open(wav_path, "rb") as f:
            return STT.audio.transcriptions.create(model="stt", file=f, language="ja").text

    text = timed("stt", stt, times)

    reply = timed("llm", lambda: LLM_.call([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]), times)

    audio = timed("tts", lambda: TTS.audio.speech.create(
        model="tts", voice=VOICE, input=reply, response_format="wav").content, times)

    Path(out_path).write_bytes(audio)
    times["total"] = time.perf_counter() - t0
    print(f"入力: {text!r}\n返答: {reply!r}")
    print("  ".join(f"{k}={v:.2f}s" for k, v in times.items()))
    return times


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
