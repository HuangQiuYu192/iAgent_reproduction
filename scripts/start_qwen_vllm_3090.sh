#!/usr/bin/env bash
# Start a private OpenAI-compatible Qwen endpoint on one RTX 3090 (24 GB).
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
API_KEY="${IAGENT_API_KEY:-iagent-local}"

if ! command -v vllm >/dev/null; then
  echo "vllm is unavailable. First run: bash scripts/setup_qwen_vllm_3090.sh" >&2
  exit 1
fi

echo "Serving ${MODEL_ID} at http://${HOST}:${PORT}/v1"
echo "This server only listens on ${HOST}; keep it private unless you deliberately change HOST."
exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" \
  --dtype half --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_MODEL_LEN}" \
  --guided-decoding-backend lm-format-enforcer
