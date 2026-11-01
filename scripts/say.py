"""1 発話を、実際の MCP につないで実行する（実機の確認用）"""
import asyncio
import sys

from vms_agent import cameras, mcp_client, service


async def main(texts: list[str]) -> None:
    cams = service.to_cameras(await cameras.load_async())
    for text in texts:
        res = await service.handle_turn(text, cams, mcp_client.show_live_view)
        print(f"[{text}] kind={res.kind} called={res.called} approx={res.approx}\n  → {res.reply}")


asyncio.run(main(sys.argv[1:]))
