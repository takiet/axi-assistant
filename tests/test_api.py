import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from vms_agent import api
from vms_agent.service import Result

CAMS = [{"id": "c1", "name": "固定カメラ", "model": "AXIS Q1656"}]
TURN_KEYS = {"turn_id", "kind", "reply", "approx", "called", "timings"}

@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    api.state["cams"], api.state["refreshed_at"] = [], 0.0
    monkeypatch.setattr(api, "turn_lock", asyncio.Lock())

def make_client(monkeypatch, handler, loader=None):
    async def fake_load():
        return CAMS
    monkeypatch.setattr(api.cameras, "load_async", loader or fake_load)
    monkeypatch.setattr(api.service, "handle_turn", handler)
    return TestClient(api.app)

async def ok_handler(*args, **kwargs):
    return Result("executed", "固定カメラを表示しました。", called=["c1"], llm_s=0.5, mcp_s=0.1)

async def boom_handler(*args, **kwargs):
    return RuntimeError("boom")

def test_error_is_not_http_error(monkeypatch):
    with make_client(monkeypatch, boom_handler) as c:
        r = c.post("/turn", json={"session_id": "t", "text": "固定カメラを表示して"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "error"
    assert body["reply"] == api.ERROR_REPLY
    assert body["called"] == []

@pytest.mark.parametrize("handler", [ok_handler, boom_handler], ids=["ok", "error"])
def test_response_shape(monkeypatch, handler):
    with make_client(monkeypatch, handler) as client:
        r = client.post("/turn", json={"session_id": "t", "text": "x"})
    body = r.json()
    assert set(body) == TURN_KEYS
    assert isinstance(body["turn_id"], str) and body["turn_id"]
    assert isinstance(body["called"], list)
    assert isinstance(body["approx"], bool)
    assert isinstance(body["timings"], dict)

@pytest.mark.parametrize("kind, reply",[
    ("executed", "固定カメラを表示しました。"),
    ("failed", "表示できませんでした。"),
])
def test_no_retry_after_mcp_called(monkeypatch, kind, reply):
    handle_calls, load_calls = [], []

    async def counting_handler(text, *args, **kwargs):
        handle_calls.append(text)
        return Result(kind, reply, called=["c1"])

    async def counting_loader():
        load_calls.append(1)
        return CAMS

    with make_client(monkeypatch, counting_handler, counting_loader) as client:
        loads_at_start = len(load_calls)
        api.state["refreshed_at"] = -1e9
        r = client.post("/turn", json={"session_id": "t", "text": "固定カメラを表示して"})

    body = r.json()
    assert body["kind"] == kind
    assert body["reply"] == reply
    assert len(handle_calls) == 1                        # 再実行していない
    assert len(load_calls) == loads_at_start

def test_retry_once_when_stale(monkeypatch):
    handle_calls, load_calls = [], []
    results = iter([Result("not_registered", "そのカメラは登録されていません。"),
                    Result("executed", "固定カメラを表示しました。", called=["c1"])])

    async def handler(text, *args, **kwargs):
        handle_calls.append(text)
        return next(results)

    async def loader():
        load_calls.append(1)
        return CAMS

    with make_client(monkeypatch, handler, loader) as client:
        loads_at_start = len(load_calls)
        api.state['refreshed_at'] = -1e9
        r = client.post("/turn", json={"session_id": "t", "text": "x"})

    assert r.json()["kind"] == "executed"
    assert len(handle_calls) == 2
    assert len(load_calls) == loads_at_start +1

def test_retry_does_not_loop(monkeypatch):
    handle_calls, load_calls = [], []

    async def handler(text, *args, **kwargs):
        handle_calls.append(text)
        return Result("not_registered", "そのカメラは登録されていません。")

    async def loader():
        load_calls.append(1)
        return CAMS

    with make_client(monkeypatch, handler, loader) as client:
        loads_at_start = len(load_calls)
        api.state["refreshed_at"] = -1e9
        r = client.post("/turn", json={"session_id": "t", "text": "x"})

    assert r.json()["kind"] == "not_registered"
    assert len(handle_calls) == 2
    assert len(load_calls) == loads_at_start + 1

def test_no_retry_when_fresh(monkeypatch):
    handle_calls, load_calls = [], []

    async def handler(text, *args, **kwargs):
        handle_calls.append(text)
        return Result("not_registered", "そのカメラは登録されていません。")

    async def loader():
        load_calls.append(1)
        return CAMS

    with make_client(monkeypatch, handler, loader) as client:
        loads_at_start = len(load_calls)
        api.state["refreshed_at"] = time.monotonic()
        r = client.post("/turn", json={"session_id": "t", "text": "x"})

    assert r.json()["kind"] == "not_registered"
    assert len(handle_calls) == 1
    assert len(load_calls) == loads_at_start


def test_start_with_no_camera(monkeypatch):
    seen = []

    async def handler(text, cams, *args, **kwargs):
        seen.append(cams)
        return Result("not_registered", "そのカメラは登録されていません。")

    async def loader():
        return []

    with make_client(monkeypatch, handler, loader) as client:
        h = client.get("/health")
        r = client.post("/turn", json={"session_id": "t", "text": "x"})

    assert h.status_code == 200 and h.json()["cameras"] == 0
    assert r.status_code == 200
    assert seen and all(c == [] for c in seen)

def test_start_when_load_fails(monkeypatch):
    async def loader():
        raise RuntimeError("mcp")

    with make_client(monkeypatch, ok_handler, loader) as client:
        h = client.get("/health")

    assert h.status_code == 200 and h.json()["cameras"] == 0
