"""1 回の kickoff で、vLLM への呼び出し回数とトークン数を測る"""
import os
import sys
import time
import urllib.request

from vms_agent import agent as vms_agent
from vms_agent import cameras

BASE = os.environ["VMS_LLM_BASE_URL"].rsplit("/v1", 1)[0]
NAMES = ("vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total")


def metrics() -> dict:
    text = urllib.request.urlopen(f"{BASE}/metrics", timeout=5).read().decode()
    out = {n: 0.0 for n in NAMES}
    for line in text.splitlines():
        for n in NAMES:
            if line.startswith(n + "{") or line.startswith(n + " "):
                out[n] += float(line.rsplit(" ", 1)[1])
    if not any(out.values()):
        print("メトリクス名が見つかりません。tokens を含む名前:",
              [l.split("{")[0] for l in text.splitlines() if "tokens" in l and not l.startswith("#")][:8])
    return out


cams = cameras.load()
for text in sys.argv[1:] or ["こんにちは"]:
    agent = vms_agent.build_agent(cams)
    before = metrics()
    t = time.perf_counter()
    result = agent.kickoff(text)
    el = time.perf_counter() - t
    after = metrics()
    d = {n.split(":")[1]: after[n] - before[n] for n in NAMES}
    gen = d["generation_tokens_total"]
    print(f"\n[{text}] {el:.2f}s  LLM呼び出し={d['request_success_total']:.0f}回  "
          f"プロンプト={d['prompt_tokens_total']:.0f}tok  生成={gen:.0f}tok  "
          f"({gen / el:.0f} tok/s)  返答={result.raw.strip()!r}")
