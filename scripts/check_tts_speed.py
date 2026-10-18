import io, statistics, time, wave
from openai import OpenAI

c = OpenAI(base_url='http://tts:6190/v1', api_key='dummy', timeout=120)
texts = ['確認します。', '入口カメラに人が一人います。', '搬入口カメラと荷捌き場のカメラを表示します。']

for i, t in enumerate(texts):
    rows = []
    for k in range(4):
        t0 = time.perf_counter()
        r = c.audio.speech.create(model='tts', voice='Ono_Anna', input=t, response_format='wav')
        el = time.perf_counter() - t0
        with wave.open(io.BytesIO(r.content)) as w:
            sec = w.getnframes() / w.getframerate()
        if k == 0:
            open(f'tests/results/faster_{i}.wav', 'wb').write(r.content)
        rows.append((el, sec))
    use = rows[1:]
    el = statistics.median(x[0] for x in use)
    sec = statistics.median(x[1] for x in use)
    print(f'{t[:16]:18s} time={el:.2f}s audio={sec:.2f}s rtf(ours)={el/sec:.2f}')
