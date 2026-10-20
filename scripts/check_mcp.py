"""MCP サーバへの接続確認: ツール一覧と get_camera_list を取得する"""
import asyncio
import json
import os
import time

from mcp import ClientSession

try:                                                         # SDK の版で関数名が違うため両対応
    from mcp.client.streamable_http import streamablehttp_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamable_http_client as http_client

URL = os.environ["MCP_URL"]                                  # 例: http://<host>:<port>/mcp


async def main():
    t0 = time.perf_counter()
    async with http_client(URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print(f"connected: {time.perf_counter() - t0:.2f}s\n")

            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
                print("    ", json.dumps(tool.inputSchema, ensure_ascii=False)[:300])

            t1 = time.perf_counter()
            result = await session.call_tool("get_camera_list", {})
            print(f"\nget_camera_list: {time.perf_counter() - t1:.2f}s")
            for c in result.content:
                print(getattr(c, "text", c)[:2000])

            await session.call_tool("show_live_view", {"camera_ids": ["68750_f3d7c461-1c6c-49e7-a565-7f3a23792cf6"]})


asyncio.run(main())
