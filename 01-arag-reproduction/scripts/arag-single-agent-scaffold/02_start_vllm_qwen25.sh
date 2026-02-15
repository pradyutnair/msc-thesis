#!/bin/bash
set -euo pipefail

# Docs:
# - vLLM OpenAI server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
# - vLLM function/tool calling: https://docs.vllm.ai/en/latest/features/tool_calling.html

MODEL_PATH="/projects/prjs1800/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
PORT="${1:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes