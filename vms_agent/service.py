"""1 回の発話を処理する。LLM は意図の判定だけ。実行・成功の判断・返答は、コードが決める"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from vms_agent import intent, mcp_client, resolver
from vms_agent.resolver import Camera

ShowFn = Callable[[list[str]], Awaitable[bool]]


@dataclass
class Result:
    kind: str                       # executed / failed / chat / list / unsupported /
                                    # ask_camera / ask_ambiguous / not_registered / error
    reply: str
    called: list[str] = field(default_factory=list)
    intent: dict | None = None
    llm_s: float = 0.0
    mcp_s: float = 0.0
    approx: bool = False            # 近似で選んだカメラを含む（聞き違いの可能性がある）


def to_cameras(cams: list[dict]) -> list[Camera]:
    return [Camera(c["id"], c["name"], c.get("model") or "") for c in cams]


def _names(cams: list[Camera]) -> str:
    return "、".join(c.name for c in cams)


async def handle_turn(text: str, cams: list[Camera], show_fn: ShowFn = mcp_client.show_live_view) -> Result:
    t0 = time.perf_counter()
    try:
        it = await asyncio.to_thread(intent.classify, text)       # 同期クライアントなので、スレッドで
    except Exception:
        return Result("error", "すみません、もう一度お願いします。")
    llm_s = time.perf_counter() - t0
    dump = it.model_dump()

    def done(kind, reply, **kw):
        return Result(kind, reply, intent=dump, llm_s=llm_s, **kw)

    if it.action == "chat":
        return done("chat", (it.reply or "").strip() or "はい。")
    if it.action == "unsupported":
        return done("unsupported", "その操作はできません。")
    if it.action == "list":
        return done("list", f"登録されているのは、{_names(cams)}です。")

    # show: 発話中のすべての言い方が、カメラに決まったときだけ実行する（一部だけの実行はしない）
    if not it.camera_phrases:
        return done("ask_camera", f"どのカメラですか。登録されているのは、{_names(cams)}です。")

    picked: list[Camera] = []
    approx = False
    for phrase in it.camera_phrases:
        r = resolver.resolve(phrase, cams)
        if r.status == "none":
            # 聞こえた言い方は、返答に含めない（音声認識の崩れた文字列を、読み上げないため）
            return done("not_registered", f"そのカメラは登録されていません。登録されているのは、{_names(cams)}です。")
        if r.status == "ambiguous":
            return done("ask_ambiguous", f"複数あります。{_names(r.cameras)}のどちらですか。")
        approx = approx or r.approx                                # 近似で選んだものがあれば、記録する
        for c in r.cameras:
            if c not in picked:
                picked.append(c)

    ids = [c.id for c in picked]                                   # ID は登録（ACS）由来のものだけ
    t1 = time.perf_counter()
    try:
        ok = await show_fn(ids)
    except Exception:
        ok = False
    mcp_s = time.perf_counter() - t1
    if not ok:
        return Result("failed", "表示できませんでした。", called=ids, intent=dump,
                      llm_s=llm_s, mcp_s=mcp_s, approx=approx)

    reply = f"{'と'.join(c.name for c in picked)}を表示しました。"
    if approx:
        reply += "違う場合は、言い直してください。"                  # 近似のときだけ、言い直しを促す
    return Result("executed", reply, called=ids, intent=dump, llm_s=llm_s, mcp_s=mcp_s, approx=approx)
