#!/usr/bin/env bash
# Serve an unquantized Qwen3-14B baseline privately on physical GPU 1.
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
API_KEY="${IAGENT_API_KEY:-iagent-qwen3-local}"
# The A40 host cannot reliably reach huggingface.co directly. Override this
# only when your site has another approved Hugging Face endpoint.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# FP16 Qwen3-14B leaves 4.82 GiB KV cache at 32K on a 46 GiB A40, while
# vLLM needs 5.00 GiB. 24K still exceeds this protocol's observed prompts.
# Two active requests are a measured throughput experiment; a pair of
# near-24K requests may still be scheduled serially by the KV cache.

exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" \
  --dtype half --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-24576}" --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_MODEL_LEN:-24576}"
