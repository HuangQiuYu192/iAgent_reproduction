#!/usr/bin/env bash
# Check a server started by start_qwen_vllm_3090.sh.
set -Eeuo pipefail

BASE_URL="${IAGENT_OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${IAGENT_API_KEY:-iagent-local}"
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/models"
echo
