# Enviromemnt

## Host
- Jetson: AGX Thor 128GB / JetPack 7.6.0
- Docker: 29.1.3 / Portainer: 2.45.1
- uv: 0.12.8

## crewai-service
- Base image: python:3.12-slim
- crewai: 1.15.23 / openai: 2.54.0 / fastapi: 0.141.1 / uvicorn: 0.54.0
- Memory: 4G / CPU: 2.0

## vllm-llm（VMS Agent 用 LLM）
- Image : nvllm-nemotron-server
- model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
- served-model-name: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
- Options: 
  --trust-remote-code --max-num-seqs 2 --max-model-len 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 1G --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 0.0.0.0 --port 6080

## Orchestrator
- Path: LiteLLM Proxy
- Model: claude opus

## Memory alloc
| Category | Amount |
|---|---|
| OS | 20GB  |
| vllm-llm | 1G |
| STT / TTS / VLM | TBD |

## Results with the test scripts
- check_vllm_openai.py tool_call: 2.25 s / no_tool: 1.31 s
- check_vllm_crewai.py tool_call: 6.60 s / no_tool: 2.92 s
- check_orchestrator.py text: 4.15 s / tool_call: 1.70 s

## TTS
- Model: Qwen3-TTS-12Hz-1.7B-CustomVoice
  - Python modules: 
    - faster-qwen3-tts==0.5.4
    - faster-qwen3-tts transformers==5.15.1 (required by faser-qwen3-tts)
  - gpu_mem_allocated=4.55GB
### Notes
- Streaming is canceled when disconnected. The lock is released once the generation of the current chunk is completed.
- Non-streaming is not canceled until the whole sentence is generated. 
- Implementation: Dedicated thread worker + threading Event. "lock" is held inside the worker (not crossing the yield)
- In Phase2, non-streaming can be also non-blocking by unifying the implementation for non-streaming to align with streaming.

