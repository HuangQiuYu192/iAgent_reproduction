#!/usr/bin/env bash
# Serve an unquantized Qwen3-14B baseline privately on physical GPU 1.
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
API_KEY="${IAGENT_API_KEY:-iagent-qwen3-local}"

exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" \
  --dtype half --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${MAX_MODEL_LEN:-32768}"
