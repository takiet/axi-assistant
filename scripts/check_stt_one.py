"""STT の疎通確認: WAV を 1 本送って文字起こしを得る"""
import sys
import time

from openai import OpenAI

BASE_URL = "http://vllm-stt:6180/v1"
client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=120)

path = sys.argv[1]
for i in range(3):   # 初回はウォームアップで遅いので、3 回測る
    start = time.perf_counter()
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="stt", file=f, language="ja",
        )
    print(f"run{i + 1}: {time.perf_counter() - start:.2f}s  {result.text!r}")
