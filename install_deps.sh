#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_HOME="${CUDA_HOME:-${HUNYUAN_CUDA_HOME:-/usr/local/cuda-12.8}}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch==2.7.1+cu128}"
TORCHVISION_PACKAGE="${TORCHVISION_PACKAGE:-torchvision==0.22.1+cu128}"
TORCHAUDIO_PACKAGE="${TORCHAUDIO_PACKAGE:-torchaudio==2.7.1+cu128}"
MODELSCOPE_PACKAGE="${MODELSCOPE_PACKAGE:-modelscope==1.37.0}"

REALESRGAN_URL="${REALESRGAN_URL:-https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth}"
U2NET_URL="${U2NET_URL:-https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx}"
U2NET_MD5="${U2NET_MD5:-60024c5c889badc19c04ad937298a77b}"
U2NET_DIR="${U2NET_DIR:-/work/models/rembg}"
MODEL_DIR="${MODEL_DIR:-/work/models/Hunyuan3D-2.1}"
DINO_DIR="${DINO_DIR:-/work/models/dinov2-giant}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  ./install_deps.sh

Environment overrides:
  HUNYUAN_CUDA_HOME=/usr/local/cuda-12.8
  TORCH_CUDA_ARCH_LIST=12.0
  PYTHON_VERSION=3.11
  VENV_DIR=.venv
  U2NET_DIR=/work/models/rembg
  DOWNLOAD_MODELS=1
  MODEL_DIR=/work/models/Hunyuan3D-2.1
  DINO_DIR=/work/models/dinov2-giant

By default this installs dependencies, builds CUDA extensions, and downloads
the RealESRGAN and rembg/u2net checkpoints. Set DOWNLOAD_MODELS=1 to also
predownload Hunyuan3D and DINOv2 weights with the ModelScope CLI.
EOF
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  log "uv not found; installing uv into the current user account"
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to install uv automatically." >&2
    exit 1
  fi

  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv install finished, but uv is still not on PATH." >&2
    exit 1
  fi
}

cuda_version() {
  "$1/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

is_cuda_128() {
  [ -x "$1/bin/nvcc" ] && [ "$(cuda_version "$1")" = "12.8" ]
}

select_cuda_home() {
  if [ -n "${HUNYUAN_CUDA_HOME:-}" ]; then
    CUDA_HOME="$HUNYUAN_CUDA_HOME"
    if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
      echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc." >&2
      exit 1
    fi
    if [ "$(cuda_version "$CUDA_HOME")" != "12.8" ] && [ "${ALLOW_CUDA_VERSION_MISMATCH:-0}" != "1" ]; then
      echo "ERROR: $CUDA_HOME is CUDA $(cuda_version "$CUDA_HOME"), but torch cu128 expects CUDA toolkit 12.8." >&2
      echo "Set ALLOW_CUDA_VERSION_MISMATCH=1 only if you know this is safe." >&2
      exit 1
    fi
    return
  fi

  for candidate in "$CUDA_HOME" /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if is_cuda_128 "$candidate"; then
      CUDA_HOME="$candidate"
      return
    fi
  done

  if [ -x "$CUDA_HOME/bin/nvcc" ] && [ "${ALLOW_CUDA_VERSION_MISMATCH:-0}" = "1" ]; then
    return
  fi

  echo "ERROR: CUDA toolkit 12.8 nvcc not found. Set HUNYUAN_CUDA_HOME=/path/to/cuda-12.8 and retry." >&2
  exit 1
}

check_tools() {
  select_cuda_home
  export CUDA_HOME
  export HUNYUAN_CUDA_HOME="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST
  export UV_LINK_MODE

  if ! command -v c++ >/dev/null 2>&1; then
    echo "ERROR: c++ compiler not found." >&2
    exit 1
  fi

  log "Using CUDA_HOME=$CUDA_HOME"
  nvcc --version | tail -n 1
}

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating virtualenv: $VENV_DIR with Python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  fi

  if [[ "$VENV_DIR" = /* ]]; then
    PYTHON="$VENV_DIR/bin/python"
  else
    PYTHON="$PWD/$VENV_DIR/bin/python"
  fi
  export PYTHON
  VENV_BIN_DIR="$(dirname "$PYTHON")"
  export VENV_BIN_DIR

  "$PYTHON" - <<PY
import sys
wanted = tuple(map(int, "$PYTHON_VERSION".split(".")[:2]))
current = sys.version_info[:2]
if current != wanted:
    raise SystemExit(f"Existing venv uses Python {current[0]}.{current[1]}, expected {wanted[0]}.{wanted[1]}. Remove {VENV_DIR} or set VENV_DIR.")
PY

  SITE_PACKAGES="$("$PYTHON" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"
}

install_python_packages() {
  log "Installing PyTorch cu128"
  uv pip install --python "$PYTHON" \
    "$TORCH_PACKAGE" "$TORCHVISION_PACKAGE" "$TORCHAUDIO_PACKAGE" \
    --extra-index-url "$PYTORCH_INDEX_URL"

  log "Installing build helpers"
  uv pip install --python "$PYTHON" wheel packaging setuptools ninja

  log "Installing project requirements"
  DS_BUILD_OPS=0 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    uv pip install --python "$PYTHON" \
      --no-build-isolation \
      --index-strategy unsafe-best-match \
      -r requirements.txt \
      "$TORCH_PACKAGE" "$TORCHVISION_PACKAGE" "$TORCHAUDIO_PACKAGE" \
      --extra-index-url "$PYTORCH_INDEX_URL"

  log "Installing ModelScope CLI"
  uv pip install --python "$PYTHON" "$MODELSCOPE_PACKAGE"
}

download_realesrgan() {
  local target="hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
  if [ -s "$target" ]; then
    log "RealESRGAN checkpoint already exists"
    return
  fi

  log "Downloading RealESRGAN checkpoint"
  mkdir -p "$(dirname "$target")"
  curl -L --fail --retry 5 --retry-delay 2 "$REALESRGAN_URL" -o "$target"
}

download_u2net() {
  local target="$U2NET_DIR/u2net.onnx"

  file_md5_ok() {
    [ -s "$1" ] && "$PYTHON" - "$1" "$U2NET_MD5" <<'PY'
import hashlib
import sys

path, expected = sys.argv[1], sys.argv[2]
h = hashlib.md5()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
raise SystemExit(0 if h.hexdigest() == expected else 1)
PY
  }

  if file_md5_ok "$target"; then
    log "rembg u2net checkpoint already exists"
    return
  fi

  log "Preparing rembg u2net checkpoint in $target"
  mkdir -p "$U2NET_DIR"

  local default_home_model="$HOME/.u2net/u2net.onnx"
  if [ ! -e "$target" ] && file_md5_ok "$default_home_model"; then
    cp "$default_home_model" "$target"
    log "Copied existing rembg u2net checkpoint from $default_home_model"
    return
  fi

  if [ ! -e "$target" ]; then
    local partial
    partial="$(find "$U2NET_DIR" -maxdepth 1 -type f -name 'tmp*' -printf '%s %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d ' ' -f2- || true)"
    if [ -n "$partial" ]; then
      mv "$partial" "$target"
    fi
  fi
  find "$U2NET_DIR" -maxdepth 1 -type f -name 'tmp*' -delete 2>/dev/null || true

  curl -L --fail --retry 5 --retry-delay 2 -C - "$U2NET_URL" -o "$target"

  if ! file_md5_ok "$target"; then
    echo "ERROR: $target failed md5 check." >&2
    exit 1
  fi
}

build_extensions() {
  log "Building custom rasterizer"
  (
    cd hy3dpaint/custom_rasterizer
    "$PYTHON" setup.py build_ext --inplace
  )

  log "Building mesh painter extension"
  (
    cd hy3dpaint/DifferentiableRenderer
    PYTHON="$PYTHON" bash compile_mesh_painter.sh
  )
}

download_models_if_requested() {
  if [ "${DOWNLOAD_MODELS:-0}" != "1" ]; then
    return
  fi

  log "Downloading Hunyuan3D weights with ModelScope"
  "$VENV_BIN_DIR/modelscope" download \
    --model Tencent-Hunyuan/Hunyuan3D-2.1 \
    --local_dir "$MODEL_DIR" \
    --include 'hunyuan3d-dit-v2-1/*' 'hunyuan3d-paintpbr-v2-1/*' \
    --max-workers "${MODELSCOPE_MAX_WORKERS:-8}"

  log "Downloading DINOv2 weights with ModelScope"
  "$VENV_BIN_DIR/modelscope" download \
    --model AI-ModelScope/dinov2-giant \
    --local_dir "$DINO_DIR" \
    --include 'config.json' 'preprocessor_config.json' 'model.safetensors' \
    --max-workers "${MODELSCOPE_DINO_MAX_WORKERS:-4}"
}

smoke_test() {
  log "Running import and CUDA smoke test"
  "$PYTHON" - <<'PY'
import sys
sys.path.insert(0, "hy3dpaint/custom_rasterizer")
sys.path.insert(0, "hy3dpaint/DifferentiableRenderer")

import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
_ = torch.randn(32, 32, device="cuda") @ torch.randn(32, 32, device="cuda")

import custom_rasterizer_kernel
import mesh_inpaint_processor
print("custom extensions import ok")
PY
}

main() {
  if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
  fi
  if [ "$#" -gt 0 ]; then
    usage >&2
    exit 2
  fi

  ensure_uv
  check_tools
  ensure_venv
  install_python_packages
  download_realesrgan
  download_u2net
  build_extensions
  download_models_if_requested
  smoke_test

  log "Dependencies are installed"
}

main "$@"
