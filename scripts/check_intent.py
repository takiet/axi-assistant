"""構造化出力で、意図とカメラの言い方が正しく出るかを確認する（思考オフ）"""
import json
import os
import time

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

client = OpenAI(base_url=os.environ["VMS_LLM_BASE_URL"], api_key="dummy", timeout=120)
MODEL = os.environ["VMS_LLM_MODEL"]


class Intent(BaseModel):
    action: Literal["show", "list", "unsupported", "chat"]
    camera_phrases: list[str] = []
    reply: str | None = None


SYSTEM = """あなたは監視カメラの音声アシスタントの「意図の判定器」です。ユーザーの発話を、次の JSON だけで答えます。
- action:
  - show: カメラの映像を見たい（表示して、見せて、映して、出して など）
  - list: 登録されているカメラの一覧を知りたい
  - unsupported: 表示以外の操作（向きを変える、録画する、ズームする など）
  - chat: あいさつや雑談
- camera_phrases: 発話に出てきたカメラの言い方を、発話の言葉のまま入れる（show のときだけ）。カメラ名を想像で補わない。
- reply: chat のときは短い返答、unsupported のときは「その操作はできません」のような短い返答。それ以外は null。
JSON 以外は何も出力しない。"""

CASES = [  # (発話, 期待 action, 期待する camera_phrases に含まれるべき語)
    ("固定カメラを表示して", "show", ["固定"]),
    ("固定カメラを見せて", "show", ["固定"]),
    ("固定カメラの映像を出して", "show", ["固定"]),
    ("ドームカメラのエリア2を表示して", "show", ["ドーム", "エリア2"]),
    ("ドームのエリア2を映して", "show", ["ドーム", "エリア2"]),
    ("ドームカメラを表示して", "show", ["ドーム"]),
    ("屋上カメラを表示して", "show", ["屋上"]),
    ("固定カメラとドームカメラを両方表示して", "show", ["固定", "ドーム"]),
    ("カメラの一覧を教えて", "list", []),
    ("どんなカメラがある？", "list", []),
    ("カメラを左に向けて", "unsupported", []),
    ("録画を始めて", "unsupported", []),
    ("こんにちは", "chat", []),
    ("ありがとう", "chat", []),
]

schema = Intent.model_json_schema()
for use in ("response_format", "guided_json"):
    print(f"\n=== 方式: {use}")
    ok = 0
    for text, want_action, want_words in CASES:
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        kwargs = {}
        if use == "response_format":
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "Intent", "schema": schema}}
        else:
            extra["guided_json"] = schema
        t = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, extra_body=extra, **kwargs,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
        except Exception as e:
            print(f"  エラー: {type(e).__name__}: {str(e)[:200]}")
            break
        el = time.perf_counter() - t
        raw = r.choices[0].message.content or ""
        try:
            got = Intent.model_validate_json(raw)
            good = got.action == want_action and all(any(w in p for p in got.camera_phrases) for w in want_words)
            ok += good
            print(f"  {'○' if good else '×'} {text:22s} {el:.2f}s tok={r.usage.completion_tokens:3d} "
                  f"{got.action:12s} {got.camera_phrases} {got.reply or ''}")
        except Exception as e:
            print(f"  × {text:22s} JSON として読めない: {raw[:120]!r}")
    print(f"  合格 {ok}/{len(CASES)}")
