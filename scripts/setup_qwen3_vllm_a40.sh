#!/usr/bin/env bash
# Create a fresh CUDA-12.4 vLLM runtime for the Qwen3-14B open-model baseline.
set -Eeuo pipefail

ENV_NAME="${IAGENT_ENV_NAME:-iagent-qwen3-a40}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/hqy/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install 'vllm==0.8.5' 'transformers>=4.51.0' 'openai>=1.40' 'pandas>=1.5'
conda run -n "${ENV_NAME}" python -m pip install -e "${ROOT_DIR}[qwen,instructrec]"
conda run -n "${ENV_NAME}" python -c 'import torch, vllm, transformers; print(torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__)'
