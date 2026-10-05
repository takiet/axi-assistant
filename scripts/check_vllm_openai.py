"""Phase 0: Jetson 上の vLLM（VMS Agent 用 LLM）の疎通と Tool Calling の確認"""
import json
import os
import sys
import time

from openai import OpenAI


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません")
    return value


BASE_URL = require_env("VMS_LLM_BASE_URL")
MODEL = require_env("VMS_LLM_MODEL")
RUNS = int(os.environ.get("CHECK_RUNS", "3"))

client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=120)

SYSTEM = (
    "あなたは VMS（映像管理システム）を操作するアシスタントです。"
    "カメラの操作が必要なときは、必ずツールを使ってください。"
    "雑談にはツールを使わず、短く答えてください。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_camera",
            "description": (
                "指定したカメラの映像を VMS に表示する。"
                "入口カメラの ID は cam-entrance-01、駐車場カメラの ID は cam-parking-01。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_id": {
                        "type": "string",
                        "description": "カメラ ID（例: cam-entrance-01）",
                    }
                },
                "required": ["camera_id"],
            },
        },
    }
]


def chat(user_text: str):
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    elapsed = time.perf_counter() - start
    return resp.choices[0], elapsed


def check_models() -> bool:
    ids = [m.id for m in client.models.list().data]
    print(f"[models] {ids}")
    if MODEL not in ids:
        print(f"  NG: {MODEL} が見つかりません（--served-model-name を確認）")
        return False
    return True


def check_tool_call() -> bool:
    choice, elapsed = chat("入口カメラを表示して")
    msg = choice.message
    print(f"[tool_call] finish_reason={choice.finish_reason}  {elapsed:.2f}s")

    if not msg.tool_calls:
        print(f"  NG: tool_calls が空です。content={msg.content!r}")
        print("  → content に <tool_call> などが含まれていれば、--tool-call-parser がモデルと合っていません")
        return False

    call = msg.tool_calls[0]
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        print(f"  NG: arguments が JSON ではありません: {call.function.arguments!r}")
        return False

    print(f"  name={call.function.name}  args={args}")
    ok = call.function.name == "show_camera" and args.get("camera_id") == "cam-entrance-01"
    print("  OK" if ok else "  NG: ツール名または引数が期待と違います")
    return ok


def check_no_tool() -> bool:
    choice, elapsed = chat("こんにちは")
    msg = choice.message
    print(f"[no_tool] finish_reason={choice.finish_reason}  {elapsed:.2f}s")
    if msg.tool_calls:
        print(f"  NG: 雑談なのにツールが呼ばれました: {msg.tool_calls[0].function.name}")
        return False
    print(f"  OK: {msg.content!r}")
    return True


def check_crewai() -> bool:
    from crewai import LLM

    llm = LLM(model=f"openai/{MODEL}", base_url=BASE_URL, api_key="dummy")
    start = time.perf_counter()
    reply = llm.call("こんにちは。一言で返してください。")
    print(f"[crewai] {time.perf_counter() - start:.2f}s  {reply!r}")
    return bool(reply)


def main() -> None:
    results = {"models": check_models()}
    if not results["models"]:
        sys.exit(1)

    tool_ok = sum(check_tool_call() for _ in range(RUNS))
    no_tool_ok = sum(check_no_tool() for _ in range(RUNS))
    results["tool_call"] = tool_ok == RUNS
    results["no_tool"] = no_tool_ok == RUNS
    results["crewai"] = check_crewai()

    print("\n=== 結果 ===")
    print(f"tool_call: {tool_ok}/{RUNS}")
    print(f"no_tool:   {no_tool_ok}/{RUNS}")
    for name, ok in results.items():
        print(f"{name:10s} {'OK' if ok else 'NG'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
