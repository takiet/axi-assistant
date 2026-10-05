"""Phase 0: CrewAI の Agent から vLLM を使い、Tool Calling が動くかを確認する"""
import os
import sys
import time

from crewai import LLM, Agent
from crewai.tools import tool

BASE_URL = os.environ["VMS_LLM_BASE_URL"]
MODEL = os.environ["VMS_LLM_MODEL"]

# ツールが呼ばれた記録（テストの判定に使う）
CALLS: list[dict] = []


@tool("show_camera")
def show_camera(camera_id: str) -> str:
    """指定したカメラの映像を VMS に表示する。
    入口カメラの ID は cam-entrance-01、駐車場カメラの ID は cam-parking-01。"""
    CALLS.append({"tool": "show_camera", "camera_id": camera_id})
    return f"{camera_id} を表示しました"


llm = LLM(
    model=f"openai/{MODEL}",
    base_url=BASE_URL,
    api_key="dummy",
    temperature=0,
)

agent = Agent(
    role="VMS オペレーター",
    goal="ユーザーの指示に従って VMS を操作し、結果を日本語で短く伝える",
    backstory="映像管理システムの操作担当。カメラ操作には必ずツールを使い、雑談にはツールを使わない。",
    tools=[show_camera],
    llm=llm,
    max_iter=3,      # ツール呼び出しのループが止まらない場合の上限
    verbose=True,    # CrewAI がモデルとどうやり取りしたかを表示
)


def run(text: str) -> tuple[list[dict], str, float]:
    CALLS.clear()
    start = time.perf_counter()
    result = agent.kickoff(text)
    elapsed = time.perf_counter() - start
    return list(CALLS), result.raw, elapsed


def main() -> None:
    ok = True

    calls, reply, elapsed = run("入口カメラを表示して")
    print(f"\n[tool_call] {elapsed:.2f}s  calls={calls}  reply={reply!r}")
    if calls != [{"tool": "show_camera", "camera_id": "cam-entrance-01"}]:
        print("  NG: 期待したツール呼び出しになっていません")
        ok = False

    calls, reply, elapsed = run("こんにちは")
    print(f"\n[no_tool] {elapsed:.2f}s  calls={calls}  reply={reply!r}")
    if calls:
        print("  NG: 雑談なのにツールが呼ばれました")
        ok = False

    print("\n=== 結果 ===", "OK" if ok else "NG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
