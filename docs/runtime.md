# Runtime Environment and Measurement Log (VMS Voice Agent)

> Last updated: 2026-10-27 (end of Phase 1)
> Legend: **[Measured]** = measured in this project / **[Config]** = current setting / **[TODO]** = not recorded or not verified yet. Fill in from the actual system.
> Related: `docs/THIRD_PARTY.md` (licenses), `docs/tts-faster-freeze.txt` (pinned dependencies of the TTS image)

---

## 1. Host

| Item | Details |
|---|---|
| Machine | Jetson AGX Thor 128GB (unified memory) |
| JetPack / L4T | [TODO] `cat /etc/nv_tegra_release`, `dpkg -l | grep nvidia-jetpack` |
| Docker / Portainer | [TODO] |
| uv (host) | [TODO] `uv --version`. Same value as `UV_VERSION` in the Dockerfile |
| Doppler | project: `vms-voice-agent` / config: `dev`. Secrets are generated into `/opt/secrets/crewai.env` and passed to `crewai-service` only |
| Development | MacBook (VS Code Remote-SSH). Used only for recording and audio conversion |
| Where things run | The app and all inference servers run on the Jetson. Only the MCP server runs on a separate Windows machine (`axis12`) |
| user | uid/gid 1000 |
| LAN IP | env:JETSON_LAN_IP |
| Livekit | livekit-agents: 1.8.9, livekit-server: 1.10.0, [TODO] -plugins-openai, -silero, -turn-detector |

### 1.1 Test Terminal

| Item | Details |
|---|---|
| Machine | Macbook Pro |
| NodeJS | v24.20.0 |

---

## 2. Service Layout

### 2.1 Networks

| Network | External access | Member services |
|---|---|---|
| `egress_net` | Yes | `crewai-service` only |
| `inference_net` | **No** (`internal: true`) | `crewai-service`, `vllm-llm`, `vllm-stt`, `tts` |
| `app_net` | No (`internal: true`) | LiveKit-related services will join later |

### 2.2 Service list

| Service | Image | Port | Role | Managed by |
|---|---|---|---|---|
| `crewai-service` | `vms/crewai-service:dev` (based on `python:3.12-slim`) | 20080 (127.0.0.1 only) | Application core (intent handling, name resolution, execution, replies) | Portainer stack `vms-voice-agent` |
| `vllm-llm` | `nvcr.io/nvidia/vllm:26.08-py3` (vLLM 0.27.1) | 6080 | Intent classification (LLM) | same |
| `vllm-stt` | `nvcr.io/nvidia/vllm:26.08-py3` | 6180 | Speech recognition | same |
| `tts` | `vms/tts-faster:dev` (based on `nvcr.io/nvidia/pytorch:26.09-py3`, built in-house) | 6190 | Speech synthesis | same |
| MCP server (external) | Custom (TypeScript, `@modelcontextprotocol/sdk ^1.12.1`) | 7080, path `/mcp` | ACS operations (Windows machine `axis12`) | Windows side |

The old image `vms/tts:dev` (plain implementation, `qwen-tts==0.1.1`, `transformers==4.57.3`) is kept as a fallback.

### 2.3 Update rules

| What changed | What to do |
|---|---|
| Code in `vms_agent/`, `scripts/` | Nothing (reflected immediately through the mount) |
| `pyproject.toml` / `uv.lock` | `uv lock --check` → rebuild the image → redeploy the stack |
| Doppler secrets | `./scripts/sync-secrets.sh` → redeploy |
| `compose.yaml` (`environment`, `command`, `image`) | Update in Portainer → **redeploy** (a restart does not apply the change). Recreating `vllm-llm` reloads the model and takes several minutes |
| TTS dependencies | Edit `docker/tts-faster/Dockerfile` → rebuild → re-check with `pip check`, `load_check.py`, `bench_faster.py` |

Do not `pip install` inside a container. Use `-i` with `docker exec` when passing a here-document (`<<EOF`).

---

## 3. Service Settings

### 3.1 `crewai-service`

| Item | Details |
|---|---|
| Dependencies | `crewai[anthropic,openai]==1.15.23`, `openai`, `fastapi`, `uvicorn` (versions in `uv.lock`). `mcp` 1.28.1, `pyyaml`, `httpx` (installed as dependencies) |
| Resources | Memory limit 4G, CPU 2.0 [Config] |
| Networks | `app_net`, `egress_net`, `inference_net` |
| Mounts | `vms_agent/`, `scripts/`, `tests/` (read-only), `tests/results/` (writable), `crewai_data` → `/data/crewai` |
| Environment variables (non-secret) | `VMS_LLM_BASE_URL` (matches `--port 6080` of `vllm-llm`), `VMS_LLM_MODEL` [TODO], `MCP_URL=http://axis12:7080/mcp`, `PYTHONPATH=/app`, `FUZZY_MIN` (default 0.6), `FUZZY_MARGIN` (default 0.1), `MCP_TIMEOUT_S` (default 5) |
| Environment variables (Doppler) | `ORCHESTRATOR_MODEL`, `ORCHESTRATOR_BASE_URL` (no trailing `/v1`), `ORCHESTRATOR_API_KEY`. **Not used in the current processing path** (only connectivity was verified in Phase 0) |
| Snapshot | `/data/crewai/cameras.json` (a copy of the camera list fetched from ACS; do not edit by hand) |
| Execution user | 1000 |
| Timeouts | LLM client (intent.py): timeout=5, max_retries=0, MCP Calling (mcp_client.py): asyncio.wait_for (5s), Camera list (cameras.py): 10s |

### 3.2 `vllm-llm`

| Item | Details |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (30B MoE with about 3B active parameters; Mamba2 + Transformer hybrid) |
| Launch options | `--trust-remote-code --max-num-seqs 2 --max-model-len 8192 --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 --kv-cache-memory-bytes 1G --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"enable_thinking": false}' --host 0.0.0.0 --port 6080` |
| **Thinking (reasoning)** | This model has **thinking ON by default**. The server default is turned OFF with `--default-chat-template-kwargs` above. See 7.4 for the reason |
| Note | Write `command` as a list in `compose.yaml` (avoids JSON quoting accidents) |

### 3.3 `vllm-stt`

| Item | Details |
|---|---|
| Model | `Qwen/Qwen3-ASR-1.7B` (`--served-model-name stt`) |
| Launch options | `--max-num-seqs 2 --max-model-len <[TODO]> --gpu-memory-utilization 0.08 --port 6180` |
| Model location | `/opt/models/hf` on the host (`HF_HOME=/models/hf`, `HF_HUB_OFFLINE=1`) |
| API | OpenAI-compatible `/v1/audio/transcriptions`, `language=ja` |
| Vocabulary hint (`prompt`) | **Not used by default.** No reliable effect was confirmed (7.1) |

### 3.4 `tts`

| Item | Details |
|---|---|
| Model | `Qwen3-TTS-12Hz-1.7B-CustomVoice`, speaker `Ono_Anna`, language `Japanese` |
| **Model location** | **`/opt/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice` (a plain directory)**, passed through `TTS_MODEL`. Specifying a repository name fails because `transformers` queries the Hub even under `HF_HUB_OFFLINE=1` |
| Speaking rate | `TTS_INSTRUCT` (the `instruct` argument) with the gist of "speak a little faster and clearly" (the Japanese instruction). The speaking rate improved ([TODO] the exact string in use) |
| Main environment variables | `TTS_MODEL`, `TTS_SPEAKER`, `TTS_INSTRUCT`, `TTS_MAX_SEQ_LEN` (default 2048), `TTS_CHUNK_SIZE` (default 4), `TTS_MAX_NEW_TOKENS` (default 512) |
| API | `GET /health`, `POST /v1/audio/speech` (OpenAI-compatible; `stream: true` streams PCM) |
| Implementation | A dedicated worker thread generates audio; the lock is acquired and released inside the worker (never held across a `yield`). Disconnects are signaled with a `threading.Event` |
| At startup | `warmup` (CUDA graph capture, about 2.2 s) → one normal synthesis → `ready.` |

### 3.5 MCP server (external, custom)

| Item | Details |
|---|---|
| Connection | Streamable HTTP, **no authentication**, `http://axis12:7080/mcp` |
| Tools | `get_server_configuration`, `get_views`, `get_camera_list`, `export_video_mp4`, `get_snapshot_jpeg`, `show_live_view` |
| Currently used | `get_camera_list` (fetch the camera list), `show_live_view` (launch live view) |
| `show_live_view` | Launches `axacs://axid.com?mode=live&cameraIds=<ID>,<ID>` on the Windows desktop. **Designed to accept multiple IDs in a single call.** Response: `{"status":"launched","url":...}` |
| Test setting | `LIVE_VIEW_DRY_RUN=1` (log only, no launch). Logs: `[tool] show_live_view camera_ids=...`, `[launch] ...ms` |
| **Safety measures (to verify)** | Validation of `camera_ids` (regex `/^[A-Za-z0-9_-]{1,64}$/` and an allow-list check) [TODO: applied?]. Windows firewall restricting inbound traffic to the Jetson's IP [TODO: applied?] |
| Note | `show_live_view` opens the window on the logged-in desktop of the user running the MCP server |

---

## 4. Memory Budget (128GB unified memory)

| Use | Setting | Measured / notes |
|---|---|---|
| OS and app layer | About 20GB reserved [policy] | |
| `vllm-llm` | `--gpu-memory-utilization 0.70`, `--kv-cache-memory-bytes 1G` | Actual usage [TODO] (KV-cache lines in `docker logs vllm-llm`, `free -h`). Weights are about 17GB (NVFP4) |
| `vllm-stt` | `--gpu-memory-utilization 0.08` (about 10GB budget) | Actual usage [TODO] |
| `tts` | `gpu_mem_allocated=4.55GB` (at ready) | Whole-process usage [TODO]. The plain implementation was 4.21GB [Measured] |
| Reference | `tegrastats` (2026-10-11 12:42): RAM 26.6GB / 125.7GB used [Measured] | Which services were loaded at that time [TODO] |
| Later (VLM etc.) | Not allocated | Phase 4 onward |

---

## 5. Reproducing the Images

### 5.1 `vms/tts-faster:dev` (`docker/tts-faster/Dockerfile`)

| Item | Details |
|---|---|
| Base | `nvcr.io/nvidia/pytorch:26.09-py3` (PyTorch `2.14.0a0+...nv26.9`, CUDA 13.4, numpy 2.1.0) |
| torchaudio | Built from source at `v2.11.0` (`BUILD_CUDA_CTC_DECODER=0 USE_CUDA=1`). The build is placed first so that its cache is reused |
| Main dependencies | `transformers==5.15.1`, `tokenizers==0.22.2`, `huggingface-hub==1.33.0`, `accelerate==1.12.0`, `qwen-tts-hf==0.1.1.post1`, `faster-qwen3-tts==0.5.4`, `fastapi==0.143.0`, `starlette==1.7.0`, `uvicorn==0.54.0`, `annotated-doc==0.0.5`. Everything else is in `docs/tts-faster-freeze.txt` |
| Constraints | `torch`, `torchaudio`, `torchvision`, `numpy` are pinned through `/etc/pip-constraints.txt` (prevents overwriting the NVIDIA builds). The base image's `PIP_CONSTRAINT=/etc/pip/constraint.txt` has no relevant pins |
| Do not install together | Upstream `qwen-tts` and `qwen-tts-hf` provide the same `qwen_tts` package, so never install both |
| `gradio` | Installed as a dependency when `faster-qwen3-tts` is installed (`gradio 6.30.0`). It is for demo UIs and is not used. Do not remove it |
| `pip check` | Only the 4 lines that come from the base image (`triton-kernels`, `cupti-python`, `flash-attn-4`, `quack-kernels`); the base image alone shows the same [Measured] |

### 5.2 Why `transformers` is pinned to 5.15.1

| Version | Result |
|---|---|
| 4.57.3 | `faster-qwen3-tts` `warmup` fails with an argument error in `StaticLayer.lazy_initialization()` (the library targets 5.x) |
| 5.16.1 – 5.19.0 | `rope_theta` disappeared from `MimiConfig` (moved into `rope_parameters`), and the compatibility shim of `qwen-tts-hf` raises `AttributeError` |
| **5.15.1** | Loading and `warmup` succeed. Matches the lower bound declared by `faster-qwen3-tts` and `qwen-tts-hf` |

When `qwen-tts-hf` or `faster-qwen3-tts` is updated, re-check the `transformers` version (`load_check.py` → `bench_faster.py`).

### 5.3 Rebuild and verify

```bash
docker build -f docker/tts-faster/Dockerfile --build-arg TRANSFORMERS_VERSION=5.15.1 \
  -t vms/tts-faster:dev --progress=plain .
docker run --rm vms/tts-faster:dev pip check
docker run --rm --runtime=nvidia --ipc=host -e HF_HUB_OFFLINE=1 \
  -v /opt/models/tts:/models/tts:ro -v $PWD/scripts:/opt/scripts \
  vms/tts-faster:dev python3 /opt/scripts/load_check.py
```

---

## 6. Application Structure (Phase 1-C)

```text
Utterance (text, or the STT output)
  → intent.classify (LLM; thinking off, structured output via response_format, 1 call)   intent and the way the camera was referred to
  → resolver.resolve (code)                                                              phrase → camera
  → mcp_client.show_live_view (code)                                                     execution; success judged by the MCP response status
  → reply (assembled by code)                                                            the LLM is never asked to claim success
```

| File | Role |
|---|---|
| `vms_agent/cameras.py` | Fetches the camera list from ACS (MCP `get_camera_list`). Excludes recorded footage (`Model == "Prerecorded video"`). Updates the snapshot and falls back to the previous copy if fetching fails |
| `vms_agent/intent.py` | Outputs `Intent` as JSON (`action`: show / list / unsupported / chat, `camera_phrases`, `reply`) |
| `vms_agent/resolver.py` | Name resolution (deterministic rules, no LLM) |
| `vms_agent/mcp_client.py` | Calls `show_live_view`. Success only when `status == "launched"` |
| `vms_agent/service.py` | `handle_turn`. Branches on the intent and returns a result kind (`kind`) and a reply |

### 6.1 Camera information policy

- **ACS is the source of truth.** There is no local settings file (`cameras.yaml`). The snapshot `/data/crewai/cameras.json` is only a stand-in for when ACS cannot be reached.
- `get_camera_list` returns at most 100 entries by default. When the camera count grows, loop with `start_index` to fetch all of them.
- Current targets (3 cameras): `固定カメラ` ("fixed camera", AXIS Q1656), `ドームカメラ` ("dome camera", AXIS M3085-V), `ドームカメラ エリア2` ("dome camera area 2", AXIS M3085-V).
- Preparing nicknames (aliases) and readings for the cameras is postponed (the ACS names are used as they are).

### 6.2 Name-resolution rules (`resolver.py`)

Stages are tried in order; the first one that decides wins.

| Stage | Description |
|---|---|
| 1 | Cameras whose name tokens (the name minus generic words such as "camera") are all contained in the phrase. Prefer the one with more tokens (more specific) |
| 2 | A word that only one camera has (e.g. `エリア2`, "area 2") appears in the phrase |
| 3 | Model number (`q1656`, `m3085`). Speech-recognition distortions (kanji numerals, "maru", "kyu", "emu") are normalized before matching |
| 4 | Digits only (4 or more digits), for when the letters of the model number were not recognized (e.g. 「九一六五六」 → fixed camera). Counted as approximate |
| 5 | Character similarity against both the name tokens and the model number. At or above `FUZZY_MIN` (default 0.6), pick those about as close as the best match. Counted as approximate |
| — | If nothing matches: "not found". If several remain equally specific: "ambiguous" |

- **Approximate matches are executed too** (the operating premise is that a wrong guess can simply be corrected by saying it again). The reply adds 「違う場合は、言い直してください。」 ("If that's wrong, please say it again.")
- Execution happens **only when every phrase in the utterance has been resolved** (never a partial execution).
- **The IDs passed to MCP come only from the registry (ACS).** There is no path by which the LLM creates an ID.
- Replies never include the phrase as heard (to avoid reading out garbled speech-recognition strings).

### 6.3 Result kinds (`kind`)

`executed` (success), `failed` (MCP failure; 「表示できませんでした」 "Could not display"), `chat`, `list`, `unsupported`, `ask_camera`, `ask_ambiguous`, `not_registered`, `error`.

---

## 7. Measurement Results

### 7.1 STT (Qwen3-ASR-1.7B)

Evaluation set: 11 clips in a quiet environment (`003` and the noisy variants were not evaluated).

| Item | Result |
|---|---|
| Processing time | 0.07–0.2 s for short utterances (0.4–0.8 s for the first call, which includes connection setup) [Measured] |
| Camera names | After the recordings were fixed, everything was recognized correctly except one case: `荷捌き場` → `にさばき場` (the reading is right but the written form is off). `入口` → `入り口` (a homophone spelling variant). Paraphrases elsewhere, e.g. `いる？` → `いますか` |
| Vocabulary hint (`prompt`) | Only one case differed (in a period that included a recording defect). **No reliable effect. Not used by default** |
| Model-number utterances | `キュー1656` → `九一六五六`, `エム3085` → `M三マル八五` (transcribed with kanji numerals and "maru"). Absorbed on the name-resolution side (6.2, stages 3 and 4) |
| Comparison | A comparison with 0.6B and Whisper large-v3-turbo was **not done** (Qwen3-ASR-1.7B was settled on and development continued) |
| Aggregates | Overall CER and camera-name accuracy: [TODO] (result files `tests/results/stt_*.json`) |

### 7.2 TTS (Qwen3-TTS-12Hz-1.7B-CustomVoice)

| Item | Plain implementation (eager) | `faster-qwen3-tts` (CUDA Graph) |
|---|---|---|
| Per frame | About 90–100 ms | 43–53 ms (benchmark; converted at 12.5 frames per second; the README says about 12, so the figure reads about 4% small) |
| RTF (processing time ÷ audio length) | 1.07–1.26 | 0.53–0.66 in the benchmark, 0.63–0.68 through the server |
| `gpu_mem_allocated` | 4.21GB | 4.37GB (benchmark), 4.55GB (server at ready) |
| Startup preparation | — | `warmup` about 2.2 s, model load about 7–8 s |

**Streaming** (`chunk_size`, longer sentence) [Measured]:

| chunk_size | Time to first audio (TTFA) | RTF |
|---|---|---|
| 2 | About 200 ms | 0.80 |
| **4 (adopted)** | About 290 ms | 0.67 |
| 8 | About 480 ms | 0.59 |

Other findings:

- Comparing `max_seq_len` 2048 and 512 shows no difference (the default 2048 is used).
- In the plain implementation, `tegrastats` showed one core at 95–100% and `nvidia-smi` GPU utilization at 45–50% (CPU launch-bound). Even with batch 4 the time was only about 1.2 times that of batch 1. Switching to 0.6B did not make it faster.
- The `RTF` in the library's log is **audio length ÷ processing time** (the inverse). Convert it before comparing.
- Subjective impression of the Japanese: acceptable. Evaluation of how camera names are read: [TODO].

**Cancellation (barge-in)** [Measured]:

| Path | Behavior |
|---|---|
| Streaming | Detects the disconnect and stops once the current chunk is done. Log: `chunks=1`, `elapsed` 0.68–1.03 s. The lock is released. Five consecutive disconnects do not pile up |
| Non-streaming | Generation does not stop on disconnect, and the lock is held for the remaining generation time (about 6 s for a long sentence). To be fixed in Phase 2 by unifying the internals on streaming generation |

### 7.3 `vllm-llm` (intent classification)

| Item | Result |
|---|---|
| Structured output | 14/14 with both `response_format` (JSON schema) and `guided_json` (paraphrases, unsupported operations, chit-chat included). 0.5–0.8 s per call, 25–43 generated tokens [Measured] |
| Generation speed | About 55–67 tokens/s (measured with the tool-calling approach) |
| Adopted | `response_format` (the OpenAI standard) |

### 7.4 Effect of thinking (reasoning) [Measured]

| Condition | 「こんにちは」 ("Hello") |
|---|---|
| Default (thinking ON) | 2.36 s, 113 tokens (`reasoning` was 351 characters; the body was a short reply) |
| Thinking OFF | 0.32 s, 12 tokens |

- `--reasoning-parser nemotron_v3` separates the thinking into another field, so it is invisible if you only look at the reply.
- With thinking ON and the Agent + MCP tool-calling approach, one call took 4.9–13 s (271–885 generated tokens).
- Setting the server default to `enable_thinking: false` improved this to about 0.5–0.7 s (9–14 generated tokens).

### 7.5 MCP

| Condition | `show_live_view` duration |
|---|---|
| `LIVE_VIEW_DRY_RUN=1` | About 0.07–0.09 s (the MCP communication itself) |
| Real launch | About 0.56–0.63 s (wait inside `launchLiveView`). Which path was used (`spawn` or `open`): [TODO] |

### 7.6 Per-stage latency (voice end-to-end, through TTS, MCP really launching) [Measured]

| Reply length | stt | llm | mcp | tts | total |
|---|---|---|---|---|---|
| 6 characters (chit-chat) | 0.07 | 0.51 | 0 | 1.62 | 2.20 |
| 10 characters (unsupported operation) | 0.12 | 0.58 | 0 | 1.69 | 2.39 |
| 13 characters (show one camera) | 0.18 | 0.71 | 0.62 | 1.39 | 2.91 |
| 20 characters (area 2) | 0.20 | 0.67 | 0.61 | 2.64 | 4.12 |
| 34 characters (list) | 0.17 | 0.48 | 0 | 3.77 | 4.42 |
| 36 characters (clarifying question) | 0.20 | 0.57 | 0 | 4.32 | 5.08 |
| 55 characters (not registered) | 0.16 | 0.63 | 0 | 5.79 | 6.57 |

- Overall median 3.72 s, maximum 6.57 s (10 clips).
- **The bottleneck is TTS, roughly proportional to the reply length** (about 0.1 s per character plus a fixed cost).
- STT plus intent classification (excluding TTS and MCP): median 0.76 s, maximum 0.94 s (10 clips).
- Streaming in Phase 2 should bring the time to first audio down to about 1.2–1.7 s [estimate].

(Reference) The old Phase 1-B chit-chat path with the Anthropic LLM: stt 0.07–0.16, llm 0.38–0.40, tts 0.92–2.19, total 1.48–2.62 s (median excluding the first call).

---

## 8. Tests

| Test | Content | Result |
|---|---|---|
| `tests/test_resolver.py` | Unit tests for name resolution (names, model numbers, digits only, approximate, not found; no LLM and no MCP) | 28/28 |
| `scripts/run_cases.py` (`tests/expected/vms_cases.yaml`) | Utterance → `handle_turn`. Checks the expected `kind` and the IDs that were called. Also covers an MCP failure response and an exception | 34/34 (median 0.57 s without MCP, 1.15 s with MCP) |
| `scripts/run_audio_cases.py` | WAV → STT → `handle_turn` → TTS | 10/10 (`tests/audio/vms_001.wav`–`vms_010.wav`) |
| `scripts/check_barge_in.py` | TTS disconnect and lock release | Passed (7.2) |

- Because `temperature=0`, repeating the same input gives the same result. **Measure accuracy by increasing the variety of utterances.**
- Expected values are written with the camera **names** (converted to IDs at run time). IDs are never hard-coded.
- Run: `docker exec crewai-service python scripts/run_cases.py [--live]` (`--live` calls MCP; start the MCP server with `LIVE_VIEW_DRY_RUN=1`).

---

## 9. Design Decision Log

| Decision | Reason |
|---|---|
| Postpone moving to Kubernetes (kind, k3s) | It is unclear whether the Thor GPU can be used inside kind. k3s has a shorter path, but device-plugin support is still unstable. Continue with Compose + Portainer |
| STT: Qwen3-ASR-1.7B | Accuracy and speed for Japanese are sufficient. The comparison was skipped |
| TTS: Qwen3-TTS + `faster-qwen3-tts` | The plain implementation has an RTF above 1; CUDA Graph brings it to about 0.6. Other candidates (Riva Magpie, AivisSpeech, VOICEVOX) were only investigated |
| Pin `transformers` to 5.15.1 | 4.57.3 makes the `faster-qwen3-tts` `warmup` fail, and 5.16 and later make the compatibility shim of `qwen-tts-hf` fail at load. Details in 5.2 |
| Framework | CrewAI first. The operation path, however, is plain code (a set of functions that can later be moved onto a `Flow`) |
| **Replaced the Agent + MCP tool-calling approach with structured output** | With thinking off, the model sometimes answered 「表示しました」 ("Displayed") **without calling the tool** (with both the old and new prompts). The decision to call a tool itself fails without thinking. The design now never lets the LLM judge success |
| ACS is the source of truth for camera information | Avoids double management. Aliases and readings are revisited later |
| Approximate matches are executed automatically | Because the operating premise is that a mistake can be corrected by saying it again. For approximate matches the reply prompts the user to say it again |
| No support for phrasing like "both" or "all" | Only utterances that name several cameras explicitly are supported |
| OpenJev (decision models) not adopted | Several unrelated projects share the name, and Japanese support and license terms are unclear. With so few cameras, rules are enough |

What was learned about CrewAI `Agent` + MCP (`mcps=[MCPServerHTTP(...)]`) (recorded because it may be used from Phase 4 on):

- The tool name seen by the LLM gets a prefix (e.g. `axis12_7080_mcp_show_live_view`). `tool_filter` works on the original name (`show_live_view`).
- **If you reuse one `Agent` and call `kickoff` repeatedly, the tools accumulate with every call.** Create an `Agent` per request.
- Signatures: `MCPServerHTTP(*, url, headers, streamable=True, tool_filter, cache_tools_list)`, `create_static_tool_filter(allowed_tool_names, blocked_tool_names)`.

---

## 10. Known Issues and Limitations

| Item | Details | Response |
|---|---|---|
| TTS is slow | Proportional to the reply length (about 0.1 s per character). Long guidance messages take 4–6 s | Shorten replies; streaming in Phase 2 |
| Disconnect on non-streaming TTS | Generation does not stop; the lock is held for about 6 s | Unify the internals on streaming in Phase 2 |
| Approximate matching | Matching by reading (kana) is not supported (「こてい」 vs 「固定」). The speech-recognition error where `Q` becomes `九` is caught by digits (counted as approximate) | Add readings when needed |
| MCP real launch | About 0.6 s | Acceptable. It has the advantage that a failure can be detected |
| Real-device multi-display check | Simultaneous display with several IDs on the real device [TODO: not verified] | `scripts/check_multi_view.py`, `scripts/say.py` |
| STT evaluation scope | Quiet environment only. Noise and other speakers not evaluated | Phase 2 onward |
| `pip check` | 4 lines that come from the base image | Leave as they are |
| `gradio` | Present in the TTS image as a dependency | Accepted |

---

## 11. Phase Progress

| Phase | Status |
|---|---|
| 0 Environment and connectivity | Done |
| 1-A STT | Done (Qwen3-ASR-1.7B) |
| 1-B STT → LLM → TTS | Done (TTS sped up, cancellation verified) |
| 1-C Voice → intent → MCP → reply | Done (except the real-device multi-display check) |
| 2 Real time (LiveKit, streaming, barge-in) | Not started. The connection framework and the first terminal still need to be chosen |
| 3 onward | Not started |

---

## 12. Unfilled / Unverified Items (summary)

- Host: versions of JetPack / Docker / Portainer / uv
- The value of `VMS_LLM_MODEL`, `--max-model-len` of `vllm-stt`, the actual `TTS_INSTRUCT` string
- Actual memory usage of each service (`free -h` before and after, `docker logs vllm-llm`)
- Whether the MCP server safety measures (ID validation, firewall) are applied
- Simultaneous display of several cameras on the real device (count, layout, upper limit)
- Which path `launchLiveView` takes (`spawn` or `open`)
- STT aggregates (CER, camera-name accuracy)
- How the TTS actually reads camera names (listening impressions)
- The "to verify" items in `docs/THIRD_PARTY.md` (e.g. the license of Qwen3-ASR)
