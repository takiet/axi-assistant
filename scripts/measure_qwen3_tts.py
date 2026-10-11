import gc
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

TEXTS = [
    "こんにちは。これは一回目の音声生成テストです。",
    "Jetson Thor 上で、同じモデルを再利用して音声を生成します。",
    "三回目の生成では、メモリ使用量と処理時間の変化を確認します。",
]

SPEAKER = "Vivian"
LANGUAGE = "Japanese"
INSTRUCT = "自然で落ち着いた話し方で。"


def read_proc_status():
    result = {}

    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                result[key] = value.strip()

    return result


def read_system_memory():
    result = {}

    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                key, value = line.split(":", 1)
                result[key] = value.strip()

    return result


def print_memory(stage):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    process = read_proc_status()
    system = read_system_memory()

    print(f"\n--- {stage} ---")
    print(f"CUDA allocated : {allocated / 1024**3:.3f} GiB")
    print(f"CUDA reserved  : {reserved / 1024**3:.3f} GiB")
    print(f"CUDA peak      : {peak / 1024**3:.3f} GiB")
    print(f"Process RSS    : {process.get('VmRSS', 'N/A')}")
    print(f"Process peak RSS: {process.get('VmHWM', 'N/A')}")
    print(f"System available: {system.get('MemAvailable', 'N/A')}")


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA device:", torch.cuda.get_device_name(0))

    print("\nLoading model...")

    start = time.perf_counter()

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize()

    print(f"Model load time: {time.perf_counter() - start:.3f} sec")
    print_memory("After model loading")

    results = []

    for i, text in enumerate(TEXTS, start=1):
        # 各生成のピークメモリを個別に計測
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=LANGUAGE,
            speaker=SPEAKER,
            instruct=INSTRUCT,
        )

        torch.cuda.synchronize()

        generation_time = time.perf_counter() - start

        wav = wavs[0]
        audio_duration = len(wav) / sample_rate
        rtf = generation_time / audio_duration

        filename = f"continuous_{i}.wav"
        sf.write(filename, wav, sample_rate)

        print(f"\n===== Generation {i} =====")
        print(f"Generation time: {generation_time:.3f} sec")
        print(f"Audio duration : {audio_duration:.3f} sec")
        print(f"RTF            : {rtf:.3f}")
        print(f"Saved          : {filename}")

        print_memory(f"After generation {i}")

        results.append({
            "iteration": i,
            "generation_time": generation_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        })

        # WAV 書き込み後、不要になった変数を解放
        del wavs, wav
        gc.collect()

    print("\n===== Summary =====")
    print(f"{'Run':>4} {'Generation(s)':>14} {'Audio(s)':>10} {'RTF':>8}")

    for r in results:
        print(
            f"{r['iteration']:>4} "
            f"{r['generation_time']:>14.3f} "
            f"{r['audio_duration']:>10.3f} "
            f"{r['rtf']:>8.3f}"
        )


if __name__ == "__main__":
    main()
