"""vms_cases.yaml を handle_turn に通して、合否と遅延を出す"""
import asyncio
import statistics
import sys
from pathlib import Path

import yaml

from vms_agent import cameras, mcp_client, service

CASES = yaml.safe_load(Path("tests/expected/vms_cases.yaml").read_text())["cases"]

async def main(live: bool) -> int:
    cams = service.to_cameras(await cameras.load_async())
    by_name = {c.name: c.id for c in cams}
    print(f"cameras: {[c.name for c in cams]}   mode={'live（MCP を呼ぶ）' if live else 'fake（MCP は呼ばない）'}\n")

    async def fake_show(ids):
        return True

    ok_n, times = 0, []
    for case in CASES:
        res = await service.handle_turn(case["say"], cams, mcp_client.show_live_view if live else fake_show)
        want = {by_name[n] for n in case.get("cameras", [])}
        called = set(res.called)
        good = res.kind == case["kind"] and called == want
        ok_n += good
        times.append(res.llm_s + res.mcp_s)
        print(f"{'○' if good else '×'} {case['say']:30s} llm={res.llm_s:.2f}s mcp={res.mcp_s:.2f}s  {res.kind:14s} {res.reply}")
        #print(f"{'○' if good else '×'} {case['say']:30s} {res.llm_s + res.mcp_s:.2f}s  {res.kind:14s} {res.reply}")
        if not good:
            print(f"    期待: kind={case['kind']} cameras={sorted(case.get('cameras', []))}")
            print(f"    実際: called={sorted(called)}  intent={res.intent}")

    async def failing_show(ids):
        return False

    async def raising_show(ids):
        raise RuntimeError("MCP down")

    print("\nMCP が失敗した場合:")
    for show, label in ((failing_show, "失敗の応答"), (raising_show, "例外")):
        res = await service.handle_turn("固定カメラを表示して", cams, show)
        good = res.kind == "failed"
        ok_n += good
        print(f"{'○' if good else '×'} MCP {label} → kind={res.kind} reply={res.reply}")

    total = len(CASES) + 2                   # 追加した 2 件も、合格数に数える
    print(f"\n合格 {ok_n}/{total}   中央値 {statistics.median(times):.2f}s   最大 {max(times):.2f}s")
    return 0 if ok_n == total else 1

sys.exit(asyncio.run(main("--live" in sys.argv)))
