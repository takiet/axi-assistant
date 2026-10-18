"""TTS のバッチ実験: 同じ文を 1 / 2 / 4 本まとめて合成し、時間の増え方を見る

目的: 遅さの原因が「GPU の演算量」か「呼び出しのオーバーヘッド（CPU 側）」かを切り分ける
  - 1 本と 4 本で時間がほぼ同じ  → オーバーヘッド支配（GPU に余裕がある）
  - 本数に比例して増える          → 演算支配
"""
import os
import statistics
import sys
import time

import torch
from qwen_tts import Qwen3TTSModel

MODEL = os.environ["TTS_MODEL"]                       # コンテナの環境変数をそのまま使う
SPEAKER = os.environ.get("TTS_SPEAKER", "Ono_Anna")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "Japanese")
TEXT = os.environ.get("BENCH_TEXT", "入口カメラに人が一人います。")
BATCHES = [int(x) for x in os.environ.get("BENCH_BATCHES", "1,2,4").split(",")]
REPEAT = int(os.environ.get("BENCH_REPEAT", "3"))
PROFILE = os.environ.get("BENCH_PROFILE") == "1"       # 1 にすると、GPU カーネル数を数える

COMPILE = os.environ.get("BENCH_COMPILE")        # 例: "reduce-overhead" / "default"

def find_predictor(model):
    """コード予測器のモジュールを探す（属性名は版で違うので、名前から探す）"""
    for name, m in model.model.named_modules():
        if m.__class__.__name__.lower().endswith("codepredictor") or "code_predictor" in name.lower():
            return name, m
    return None, None

# ウォームアップを増やす（形状ごとにコンパイルが走るため）
WARMUP = int(os.environ.get("BENCH_WARMUP", "2" if not COMPILE else "6"))

def to_len(wav) -> int:
    """音声のサンプル数（numpy でも tensor でも対応）"""
    return int(wav.shape[-1]) if hasattr(wav, "shape") else len(wav)


def synth(model, n: int):
    # ★ ここを、動作確認済みの呼び出しに合わせる（メソッド名・引数名）
    return model.generate_custom_voice(
        text=[TEXT] * n,
        language=[LANGUAGE] * n,
        speaker=[SPEAKER] * n,
    )


def timed(model, n: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    wavs, sr = synth(model, n)
    torch.cuda.synchronize()
    return time.perf_counter() - start, wavs, sr


def main() -> None:
    print(f"model={MODEL}\ntext={TEXT!r} ({len(TEXT)}文字)  batches={BATCHES}  repeat={REPEAT}")
    model = Qwen3TTSModel.from_pretrained(                 # ★ server.py と同じ引数に合わせる
        MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    # model を作った直後
    if COMPILE:
        name, pred = find_predictor(model)
        print("code predictor:", name, type(pred).__name__)
        pred.model.forward = torch.compile(pred.model.forward, mode=COMPILE, fullgraph=False)

    # ウォームアップ（結果には含めない）
    for _ in range(2):
        synth(model, 1)
    print(f"warmup done  gpu_mem_allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB")

    rows = []
    for n in BATCHES:
        times, audio_sec = [], 0.0
        for _ in range(REPEAT):
            try:
                elapsed, wavs, sr = timed(model, n)
            except (TypeError, ValueError, RuntimeError) as e:
                print(f"\n[batch={n}] 失敗: {type(e).__name__}: {e}")
                print("  → バッチ（リスト）で渡せない場合は、署名を確認: inspect.signature(generate_custom_voice)")
                sys.exit(1)
            times.append(elapsed)
            audio_sec = sum(to_len(w) for w in wavs) / sr   # バッチ全体の音声の長さ（秒）
        med = statistics.median(times)
        rows.append((n, med, min(times), max(times), audio_sec))
        print(f"batch={n}: median={med:.2f}s (min={min(times):.2f} max={max(times):.2f})"
              f"  audio_total={audio_sec:.2f}s  rtf_total={med / audio_sec:.2f}"
              f"  time_per_item={med / n:.2f}s")

    # 1 本に対する相対値
    base = rows[0][1]
    print("\n=== 1 本に対する比 ===")
    for n, med, *_ in rows:
        print(f"batch={n}: 時間 x{med / base:.2f}   （本数に比例なら x{n:.2f}、オーバーヘッド支配なら x1.0 付近）")

    if PROFILE:
        profile_once(model)


def profile_once(model) -> None:
    """1 本の合成で、GPU カーネルが何回発行されたかを数える（ベストエフォート）"""
    from torch.profiler import ProfilerActivity, profile
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            synth(model, 1)
            torch.cuda.synchronize()
        events = prof.events()
        cuda = [e for e in events if str(getattr(e, "device_type", "")).endswith("CUDA")]
        total_us = sum(getattr(e, "device_time", None) or getattr(e, "cuda_time", 0) or 0 for e in cuda)
        print(f"\n=== プロファイル（1 本） ===")
        print(f"GPU 上のイベント数: {len(cuda)}")
        if cuda:
            print(f"GPU 上の実行時間の合計: {total_us / 1e6:.2f}s   平均: {total_us / len(cuda):.1f}us / イベント")
        print("  → 数千〜数万のイベントで、平均が数十 us 以下なら「小さな呼び出しが大量」（オーバーヘッド支配）")
    except Exception as e:                                    # プロファイラの API は版で変わる
        print(f"\nプロファイルは取得できませんでした: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
