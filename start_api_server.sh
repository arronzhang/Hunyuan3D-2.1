#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_HOME="${HUNYUAN_CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PWD/.venv/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$PWD:$PWD/hy3dpaint:$PWD/hy3dshape:${PYTHONPATH:-}"
export DINO_CKPT_PATH="${DINO_CKPT_PATH:-/work/models/dinov2-giant}"
export U2NET_HOME="${U2NET_HOME:-/work/models/rembg}"

exec .venv/bin/python api_server.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8081}" \
  "$@"
