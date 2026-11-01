"""発話 → 意図とカメラの言い方（LLM、思考オフ、構造化出力）"""
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel


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
- camera_phrases: 発話に出てきたカメラの言い方を、発話の言葉のまま入れる（show のときだけ）。複数のカメラが挙がっていれば、1 つずつ別々の要素にする。カメラ名を想像で補わない。  # ★
- reply: chat のときは短い返答、unsupported のときは「その操作はできません」のような短い返答。それ以外は null。
JSON 以外は何も出力しない。"""

MODEL = os.environ["VMS_LLM_MODEL"]
_client = OpenAI(base_url=os.environ["VMS_LLM_BASE_URL"], api_key="dummy", timeout=5)
_SCHEMA = Intent.model_json_schema()


def classify(text: str) -> Intent:
    r = _client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        response_format={"type": "json_schema", "json_schema": {"name": "Intent", "schema": _SCHEMA}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return Intent.model_validate_json(r.choices[0].message.content or "")
