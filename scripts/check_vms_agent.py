"""VMS Agent の動作確認（テキスト入力）。Tool の呼び出しは、MCP サーバ側のログで確認する"""
import sys
import time

from vms_agent import agent as vms_agent
from vms_agent import cameras

cams = cameras.load()
print(f"cameras: {len(cams)} 台")
for c in cams:
    print(" ", c["name"], c["id"])

agent = vms_agent.build_agent(cams)

texts = sys.argv[1:] or ["固定カメラを表示して"]
for text in texts:
    t = time.perf_counter()
    result = agent.kickoff(text)
    print(f"\n[入力] {text}\n[返答] {result.raw!r}\n[時間] {time.perf_counter() - t:.2f}s\n")
