#!/usr/bin/env bash
set -euo pipefail

# Run this only after the local vLLM service passes scripts/check_qwen_vllm.sh.
# Results are JSONL checkpoints: rerunning this command resumes successful rows.
: "${IAGENT_API_KEY:?Set IAGENT_API_KEY first}"
export IAGENT_OPENAI_BASE_URL="${IAGENT_OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"

python -m iagent_reproduction.run_official_protocol \
  --agent static \
  --model Qwen/Qwen2.5-7B-Instruct \
  --all \
  --output outputs/official_protocol/books_static_qwen.jsonl
