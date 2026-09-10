#!/usr/bin/env bash
# Install a CUDA-12.4 vLLM runtime for one 46 GB NVIDIA A40.
set -Eeuo pipefail

ENV_NAME="${IAGENT_ENV_NAME:-iagent-qwen-a40}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi was not found. Run this on the A40 server." >&2
  exit 1
fi
if ! command -v conda >/dev/null; then
  echo "conda was not found. Source your Miniconda conda.sh, then retry." >&2
  exit 1
fi

echo "Detected GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# vLLM 0.8.5 ships CUDA-12.4 binaries, matching the A40 server's CUDA 12.4
# driver. Use a fresh environment: vLLM CUDA extensions are binary-sensitive.
conda run -n "${ENV_NAME}" python -m pip install "vllm==${VLLM_VERSION}" "openai>=1.40" "pandas>=1.5"
conda run -n "${ENV_NAME}" python -m pip install -e "${ROOT_DIR}[qwen,instructrec]"
conda run -n "${ENV_NAME}" python -c "import torch, vllm; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('vllm:', vllm.__version__)"

cat <<EOF

Installed environment: ${ENV_NAME}
Suggested one-A40 launch (physical GPU 1 is empty at setup time):
  conda activate ${ENV_NAME}
  export IAGENT_API_KEY='choose-a-local-secret'
  CUDA_VISIBLE_DEVICES=1 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.85 bash scripts/start_qwen_vllm_3090.sh
EOF
