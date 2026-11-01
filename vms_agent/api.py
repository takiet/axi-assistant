# 構成の骨子（実物の関数名・引数に合わせること）
import asyncio, time, uuid, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from vms_agent import cameras, service

log = logging.getLogger("turn")
state = {"cams": [], "refreshed_at": 0.0}
turn_lock = asyncio.Lock()
ERROR_REPLY = "すみません、もう一度お願いします。"

async def refresh():
    cams = service.to_cameras(await cameras.load_async())
    if cams:
        state["cams"], state["refreshed_at"] = cams, time.monotonic()

async def refresher():
    while True:
        await asyncio.sleep(30 if not state["cams"] else 300)
        try: await refresh()
        except Exception: log.exception("camera refresh failed")

@asynccontextmanager
async def lifespan(app):
    try: await refresh()
    except Exception: log.exception("initial camera load failed")
    task = asyncio.create_task(refresher())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

class TurnIn(BaseModel):
    session_id: str
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "cameras": len(state["cams"])}

@app.post("/turn")
async def turn(req: TurnIn):
    turn_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    async def work():
        async with turn_lock:
            r = await service.handle_turn(req.text, state["cams"])
            if r.kind == "not_registered" and time.monotonic() - state["refreshed_at"] > 30:
                await refresh()                 # 実行前の状態のときだけ、再取得して 1 回だけやり直す
                r = await service.handle_turn(req.text, state["cams"])
            return r
    try:
        r = await asyncio.shield(asyncio.create_task(work()))
        out = {"turn_id": turn_id, "kind": r.kind, "reply": r.reply, "approx": r.approx,
               "called": r.called, "timings": {"llm_s": r.llm_s, "mcp_s": r.mcp_s}}
    except Exception:
        log.exception("turn failed turn_id=%s", turn_id)
        out = {"turn_id": turn_id, "kind": "error", "reply": ERROR_REPLY, "approx": False,
               "called": [], "timings": {}}
    log.info("turn_id=%s session=%s kind=%s called=%s total=%.2fs",
             turn_id, req.session_id, out["kind"], out["called"], time.perf_counter() - t0)
    return out
