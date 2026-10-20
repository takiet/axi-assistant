"""ACS（MCP 経由）からカメラ一覧を取得する。ACS が正。ローカルは派生物（スナップショット）"""
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession

try:                                                       # SDK の版で関数名が違うため両対応
    from mcp.client.streamable_http import streamablehttp_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamable_http_client as http_client

MCP_URL = os.environ["MCP_URL"]
SNAPSHOT = Path(os.environ.get("CAMERA_SNAPSHOT", "/data/crewai/cameras.json"))
EXCLUDE_MODELS = {"Prerecorded video"}                     # 録画映像は対象外


async def _fetch() -> list[dict]:
    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            # 既定は 100 件まで。台数が増えたら start_index を進めて全件を取るループにする
            result = await session.call_tool("get_camera_list", {"number_of_elements": 1000})
            return json.loads(result.content[0].text)


def load() -> list[dict]:
    """取得してスナップショットを更新。失敗したら前回のスナップショットで代用"""
    try:
        raw = asyncio.run(_fetch())
        cams = [{"id": c["CameraId"], "name": c["Name"], "model": c.get("Model")}
                for c in raw if c.get("Model") not in EXCLUDE_MODELS]
        SNAPSHOT.write_text(json.dumps(cams, ensure_ascii=False, indent=2))
        return cams
    except Exception:
        if SNAPSHOT.exists():
            return json.loads(SNAPSHOT.read_text())
        raise


def as_table(cams: list[dict]) -> str:
    return "\n".join(f"- {c['name']}（ID: {c['id']}、型番: {c['model']}）" for c in cams)
