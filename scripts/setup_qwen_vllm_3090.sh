#!/usr/bin/env bash
# Install the reproducible local-Qwen runtime from a Jupyter/Linux terminal.
set -Eeuo pipefail

# The server that motivated this script exposes CUDA driver 12.2.  Pin a vLLM
# CUDA-12.1 wheel: newer vLLM releases may install a PyTorch CUDA runtime that
# requires a newer NVIDIA driver.
ENV_NAME="${IAGENT_ENV_NAME:-iagent-qwen-cu121}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VLLM_VERSION="${VLLM_VERSION:-0.6.3.post1}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi was not found. Run this on the GPU Jupyter server." >&2
  exit 1
fi
if ! command -v conda >/dev/null; then
  echo "conda was not found. Load your server's Conda module, then retry." >&2
  exit 1
fi

echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
if [ "${GPU_MEMORY_MB}" -lt 20000 ]; then
  echo "Warning: this script is tuned for a 24 GB RTX 3090; detected ${GPU_MEMORY_MB} MiB." >&2
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# vLLM 0.6.3's released Linux wheel is compiled for CUDA 12.1, which is
# compatible with a CUDA 12.2 driver. Keep this pin for reproducibility.
conda run -n "${ENV_NAME}" python -m pip install "vllm==${VLLM_VERSION}" "openai>=1.40" "pandas>=1.5"
conda run -n "${ENV_NAME}" python -c "import torch, vllm; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('vllm:', vllm.__version__)"

cat <<EOF

Installed environment: ${ENV_NAME}
Next, in this repository's Jupyter terminal:
  conda activate ${ENV_NAME}
  export IAGENT_API_KEY='choose-a-local-secret'
  bash scripts/start_qwen_vllm_3090.sh
EOF
