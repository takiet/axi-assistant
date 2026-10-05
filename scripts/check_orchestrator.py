"""Phase 0: Orchestrator（LiteLLM Proxy の Anthropic パススルー経由）の疎通確認"""
import os
import sys
import time

from crewai import LLM, Agent
from crewai.tools import tool


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません（/opt/secrets/crewai.env を確認）")
    return value


MODEL = require_env("ORCHESTRATOR_MODEL")
BASE_URL = require_env("ORCHESTRATOR_BASE_URL")
API_KEY = require_env("ORCHESTRATOR_API_KEY")

# キーそのものは表示しない
print(f"model={MODEL}  base_url={BASE_URL}  api_key=...{API_KEY[-4:]}")

llm = LLM(model=MODEL, base_url=BASE_URL, api_key=API_KEY)

# ツールが呼ばれた記録（判定に使う）
CALLS: list[dict] = []


@tool("route_to_vms")
def route_to_vms(reason: str) -> str:
    """カメラの表示・操作・映像の確認など、VMS に関する依頼を VMS Agent に引き渡す。
    reason には引き渡す理由を短く書く。"""
    CALLS.append({"tool": "route_to_vms", "reason": reason})
    return "VMS Agent に引き渡しました"


agent = Agent(
    role="Orchestrator",
    goal="ユーザーの依頼を判定し、VMS に関する依頼は VMS Agent に引き渡す。雑談には自分で短く答える。",
    backstory="音声アシスタントの司令塔。VMS の操作には必ず route_to_vms を使い、雑談にはツールを使わない。",
    tools=[route_to_vms],
    llm=llm,
    max_iter=3,
    verbose=True,
)


def check_text() -> bool:
    start = time.perf_counter()
    reply = llm.call("こんにちは。一言で返してください。")
    print(f"\n[text] {time.perf_counter() - start:.2f}s  {reply!r}")
    return bool(reply)


def run(text: str) -> tuple[list[dict], str, float]:
    CALLS.clear()
    start = time.perf_counter()
    result = agent.kickoff(text)
    return list(CALLS), result.raw, time.perf_counter() - start


def main() -> None:
    results = {"text": check_text()}

    calls, reply, elapsed = run("入口カメラを表示して")
    print(f"\n[tool_call] {elapsed:.2f}s  calls={calls}  reply={reply!r}")
    results["tool_call"] = len(calls) == 1 and calls[0]["tool"] == "route_to_vms"

    calls, reply, elapsed = run("こんにちは")
    print(f"\n[no_tool] {elapsed:.2f}s  calls={calls}  reply={reply!r}")
    results["no_tool"] = not calls

    print("\n=== 結果 ===")
    for name, ok in results.items():
        print(f"{name:10s} {'OK' if ok else 'NG'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
