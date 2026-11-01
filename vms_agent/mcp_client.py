"""MCP の show_live_view を呼ぶ。成功の判定は、応答の status で行う"""
import asyncio
import json
import os

from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamablehttp_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamable_http_client as http_client

MCP_URL = os.environ["MCP_URL"]
MCP_TIMEOUT_S = float(os.environ.get("MCP_TIMEOUT_S", "5"))


async def _call(camera_ids: list[str]) -> bool:
    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool("show_live_view", {"camera_ids": camera_ids})
            if getattr(result, "isError", False):
                return False
            try:
                return json.loads(result.content[0].text).get("status") == "launched"
            except (IndexError, ValueError, AttributeError):
                return False


async def show_live_view(camera_ids: list[str]) -> bool:
    try:
        return await asyncio.wait_for(_call(camera_ids), MCP_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False            # 時間切れは失敗扱い。実際には起動していても、言い直せば分かる
