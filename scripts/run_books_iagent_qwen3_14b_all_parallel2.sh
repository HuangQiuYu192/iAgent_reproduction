#!/usr/bin/env bash
set -Eeuo pipefail

: "${IAGENT_API_KEY:?Set a private local server key first}"
export IAGENT_OPENAI_BASE_URL="${IAGENT_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}"

python -m iagent_reproduction.run_official_protocol \
  --agent static \
  --model Qwen/Qwen3-14B \
  --protocol-mode strict \
  --json-mode json_schema \
  --disable-thinking \
  --dynamic-output-budget \
  --context-window 24576 \
  --max-output-tokens 16384 \
  --output-safety-tokens 256 \
  --tokenizer Qwen/Qwen3-14B \
  --workers 2 \
  --all \
  --output outputs/official_protocol/books_static_qwen3_14b_fp16_all_parallel2.jsonl
