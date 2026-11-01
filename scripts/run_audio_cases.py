"""音声ケース: WAV → STT → handle_turn →（任意で）TTS。合否と区間ごとの時間を出す"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

from vms_agent import cameras, mcp_client, service

CASES = [c for c in yaml.safe_load(Path("tests/expected/vms_cases.yaml").read_text())["cases"] if c.get("audio")]
STT = OpenAI(base_url="http://vllm-stt:6180/v1", api_key="dummy", timeout=60)
TTS = OpenAI(base_url="http://faster-tts:6280/v1", api_key="dummy", timeout=120)
OUT = Path("tests/results/audio_cases")


def transcribe(path: Path) -> str:
    with open(path, "rb") as f:
        return STT.audio.transcriptions.create(model="stt", file=f, language="ja").text


def speak(text: str, out: Path) -> float:
    t = time.perf_counter()
    r = TTS.audio.speech.create(model="tts", voice="Ono_Anna", input=text, response_format="wav")
    out.write_bytes(r.content)
    return time.perf_counter() - t


async def main(live: bool, with_tts: bool) -> int:
    cams = service.to_cameras(await cameras.load_async())
    by_name = {c.name: c.id for c in cams}
    OUT.mkdir(parents=True, exist_ok=True)

    async def fake_show(ids):
        return True

    show = mcp_client.show_live_view if live else fake_show
    transcribe(Path("tests/audio") / CASES[0]["audio"])            # ウォームアップ
    ok_n, totals = 0, []
    for case in CASES:
        t = time.perf_counter()
        heard = await asyncio.to_thread(transcribe, Path("tests/audio") / case["audio"])
        stt_s = time.perf_counter() - t
        res = await service.handle_turn(heard, cams, show)
        tts_s = await asyncio.to_thread(speak, res.reply, OUT / f"{Path(case['audio']).stem}.wav") if with_tts else 0.0

        want = {by_name[n] for n in case.get("cameras", [])}
        good = res.kind == case["kind"] and set(res.called) == want
        ok_n += good
        total = stt_s + res.llm_s + res.mcp_s + tts_s
        totals.append(total)
        print(f"{'○' if good else '×'} {case['audio']:12s} stt={stt_s:.2f} llm={res.llm_s:.2f} mcp={res.mcp_s:.2f} "
              f"tts={tts_s:.2f} 計={total:.2f}s  {res.kind}")
        print(f"      聞こえた: {heard!r}   返答: {res.reply}   approx={res.approx}")
        if not good:
            print(f"      期待: kind={case['kind']} cameras={sorted(case.get('cameras', []))}  実際 called={sorted(res.called)}")
            print(f"      intent={res.intent}")

    print(f"\n合格 {ok_n}/{len(CASES)}   中央値 {statistics.median(totals):.2f}s   最大 {max(totals):.2f}s")
    return 0 if ok_n == len(CASES) else 1


sys.exit(asyncio.run(main("--live" in sys.argv, "--tts" in sys.argv)))
