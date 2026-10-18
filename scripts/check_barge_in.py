"""barge-in（割り込み）の確認: 切断後に生成が止まり、次のリクエストが待たされないか"""
import statistics
import time

import httpx

URL = "http://faster-tts:6280/v1/audio/speech"
LONG = "入口カメラ、駐車場カメラ、搬入口カメラの三台を順に表示します。それぞれの映像を確認してから、人の有無をお伝えします。"
SHORT = "確認します。"


def payload(text, stream=False):
    return {"model": "tts", "voice": "Ono_Anna", "input": text,
            "response_format": "wav", "stream": stream}


def normal(text=SHORT, timeout=120):
    t = time.perf_counter()
    httpx.post(URL, json=payload(text), timeout=timeout).raise_for_status()
    return time.perf_counter() - t

def normal_or_stuck(text=SHORT):
    try:
        return normal(text, timeout=20)
    except httpx.ReadTimeout:
        return None

def abort_nonstream(text, after):
    """非ストリーミングで依頼し、after 秒で接続を切る"""
    try:
        httpx.post(URL, json=payload(text), timeout=httpx.Timeout(after, connect=5))
    except httpx.ReadTimeout:
        pass


def abort_stream(text):
    """ストリーミングで依頼し、最初のデータを受けたら接続を切る"""
    t = time.perf_counter()
    ttfa = None
    with httpx.stream("POST", URL, json=payload(text, True), timeout=120) as r:
        for _ in r.iter_bytes():
            ttfa = time.perf_counter() - t
            break                      # with を抜けると接続が閉じる
    return ttfa


def verdict(extra):
    return "OK" if extra <= 0.6 else ("要注意" if extra <= 2.0 else "NG（生成が止まっていない可能性）")


def main():
    normal()                                                   # ウォームアップ
    base = statistics.median(normal() for _ in range(3))
    print(f"基準: 短い依頼の単独の所要時間 = {base:.2f}s\n")

    print("[1] 非ストリーミング: 0.5 秒で切断 → 直後に短い依頼")
    abort_nonstream(LONG, 0.5)
    t1 = normal()
    print(f"    {t1:.2f}s（基準 +{t1 - base:.2f}s）  ← 生成が止まらないなら、残りの生成を待つ")
    time.sleep(10)

    print("\n[2] ストリーミング: 最初の音を受けたら切断 → 直後に短い依頼")
    ttfa = abort_stream(LONG)
    t2 = normal_or_stuck()
    if t2 is None:
        print("    20 秒以内に応答なし → NG（ロックが解放されていない）。tts を再起動してください: docker restart tts")
        return
    print(f"    最初の音まで {ttfa:.2f}s / 短い依頼 {t2:.2f}s（基準 +{t2 - base:.2f}s）→ {verdict(t2 - base)}")
    time.sleep(10)

    print("\n[3] ストリーミングを 5 回連続で切断 → 直後に短い依頼")
    for _ in range(5):
        abort_stream(LONG)
    t3 = normal_or_stuck()
    if t3 is None:
        print("    20 秒以内に応答なし → NG（ロックが解放されていない）。tts を再起動してください: docker restart tts")
        return
    print(f"    最初の音まで {ttfa:.2f}s / 短い依頼 {t3:.2f}s（基準 +{t3 - base:.2f}s）→ {verdict(t3 - base)}")
    time.sleep(10)

    print("\n[4] 切断のあとの通常の依頼が、正常に完了するか")
    r = httpx.post(URL, json=payload(LONG), timeout=120)
    open("tests/results/barge_in_after.wav", "wb").write(r.content)
    print(f"    status={r.status_code} bytes={len(r.content)}  （保存した音を聞いて、崩れがないか確認）")


if __name__ == "__main__":
    main()
