"""vLLM を直接呼び、思考（reasoning）の有無と、無効化したときのトークン数の差を見る"""
import os
import time

from openai import OpenAI

client = OpenAI(base_url=os.environ["VMS_LLM_BASE_URL"], api_key="dummy", timeout=120)
MODEL = os.environ["VMS_LLM_MODEL"]
MSGS = [
    {"role": "system", "content": "あなたは監視カメラの音声アシスタントです。返答は1文で短く。"},
    {"role": "user", "content": "こんにちは"},
]


def run(label, **extra):
    t = time.perf_counter()
    r = client.chat.completions.create(model=MODEL, messages=MSGS, temperature=0, **extra)
    el = time.perf_counter() - t
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
    print(f"[{label}] {el:.2f}s  completion_tokens={r.usage.completion_tokens}")
    print("   content  :", repr(m.content)[:160])
    print("   reasoning:", (repr(reasoning)[:160] + f"  ...（{len(reasoning)}文字）") if reasoning else None)


run("既定")
run("思考オフ", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
