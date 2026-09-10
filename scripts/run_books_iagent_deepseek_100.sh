#!/usr/bin/env bash
set -euo pipefail

# Do not put a real key in this file. Export DEEPSEEK_API_KEY in the A40 shell.
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in this shell first}"

python -m iagent_reproduction.run_official_protocol \
  --agent static \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --protocol-mode strict \
  --json-mode json_object \
  --limit 100 \
  --output outputs/official_protocol/books_static_deepseek_v4_flash_100_v2.jsonl
