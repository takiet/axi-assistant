"""Phase 1-A: STT の CER・カメラ名の正解率・速度を測る"""
import json
import os
import statistics
import sys
import time
import unicodedata
import re
import wave
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

BASE_URL = os.environ.get("STT_BASE_URL", "http://vllm-stt:6180/v1")
MODEL = os.environ.get("STT_MODEL", "stt")
LANGUAGE = os.environ.get("STT_LANGUAGE", "ja")
LABEL = os.environ.get("STT_LABEL", "qwen3-asr-1.7b")           # 例: qwen3-asr-1.7b（結果ファイル名に使う）
PROMPT = os.environ.get("STT_PROMPT")         # Step 5（keyterm）で使う
TEMPERATURE = float(os.environ.get("STT_TEMPERATURE", "0"))
if not LABEL:
    sys.exit("STT_LABEL を指定してください（例: STT_LABEL=qwen3-asr-1.7b）")

client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=120)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s、。，．,.!?！？「」『』]", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def duration_sec(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def transcribe(path: Path) -> tuple[str, float]:
    kwargs = {"model": MODEL, "language": LANGUAGE, "temperature": TEMPERATURE}
    if PROMPT:
        kwargs["prompt"] = PROMPT
    start = time.perf_counter()
    with open(path, "rb") as f:
        res = client.audio.transcriptions.create(file=f, **kwargs)
    return res.text, time.perf_counter() - start


def camera_names(cameras: dict, camera_id: str) -> list[str]:
    cam = cameras[camera_id]
    return [cam["name"], *cam.get("aliases", [])]


def main() -> None:
    cases = yaml.safe_load(Path("tests/expected/cases.yaml").read_text())["cases"]
    cameras = yaml.safe_load(Path("config/cameras.yaml").read_text())

    # 評価対象の一覧を作る
    items = []
    for case_id, case in cases.items():
        camera_id = (case.get("expect", {}).get("args") or {}).get("camera_id")
        for name in case.get("audio") or [f"{case_id}.wav"]:
            items.append((case_id, name, case["transcript"], camera_id))

    # ウォームアップ（1 回目は遅いので、結果に含めない）
    transcribe(Path("tests/audio") / items[0][1])

    rows = []
    for case_id, name, ref, camera_id in items:
        path = Path("tests/audio") / name
        hyp, elapsed = transcribe(path)
        dur = duration_sec(path)
        row = {
            "case": case_id, "file": name,
            "condition": "noisy" if "noisy" in name else "quiet",
            "ref": ref, "hyp": hyp,
            "cer": cer(ref, hyp),
            "latency": elapsed, "rtf": elapsed / dur,
            "camera_id": camera_id,
            "camera_ok": None,
        }
        if camera_id:
            row["camera_ok"] = any(
                normalize(n) in normalize(hyp) for n in camera_names(cameras, camera_id)
            )
        rows.append(row)
        flag = "" if row["camera_ok"] in (None, True) else "  ← カメラ名NG"
        print(f"{name:40s} CER={row['cer']:.3f} {elapsed:.2f}s  {hyp!r}{flag}")

    # 集計
    def mean_of(key, subset):
        return statistics.mean(r[key] for r in subset) if subset else float("nan")

    cam_rows = [r for r in rows if r["camera_ok"] is not None]
    summary = {
        "label": LABEL, "n": len(rows),
        "cer_mean": mean_of("cer", rows),
        "cer_quiet": mean_of("cer", [r for r in rows if r["condition"] == "quiet"]),
        "cer_noisy": mean_of("cer", [r for r in rows if r["condition"] == "noisy"]),
        "camera_acc": (sum(r["camera_ok"] for r in cam_rows) / len(cam_rows)) if cam_rows else None,
        "latency_mean": mean_of("latency", rows),
        "rtf_mean": mean_of("rtf", rows),
    }
    print("\n=== 結果 ===")
    for k, v in summary.items():
        print(f"{k:14s} {v}")

    out = Path("tests/results") / f"stt_{LABEL}_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(f"保存しました: {out}")


if __name__ == "__main__":
    main()
