"""Thor で CUDA graphs / torch.compile が動くかの確認（小さな層を多数並べた、起動コスト支配のモデル）"""
import time

import torch
import torch.nn as nn

dev = "cuda"
model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(24)]).to(dev).half().eval()
x = torch.randn(1, 1024, device=dev, dtype=torch.half)


def bench(fn, n=200):
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


with torch.inference_mode():
    print(f"eager:      {bench(model):.3f} ms")

    # (a) 手動の CUDA graph
    try:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                model(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            model(x)
        print(f"cuda graph: {bench(lambda _: g.replay()):.3f} ms")
    except Exception as e:
        print("cuda graph 失敗:", type(e).__name__, e)

    # (b) torch.compile（reduce-overhead は CUDA graphs を内包）
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        t = time.perf_counter()
        compiled(x)
        print(f"compile 初回: {time.perf_counter() - t:.1f}s")
        print(f"compiled:   {bench(compiled):.3f} ms")
    except Exception as e:
        print("compile 失敗:", type(e).__name__, str(e)[:300])
