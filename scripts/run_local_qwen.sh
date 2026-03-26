#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

HAVE_MODULE=false
if command -v module >/dev/null 2>&1; then
  HAVE_MODULE=true
  if ! module load python/3.10.3; then
    echo "WARNING: could not load python/3.10.3 module; using existing .venv python." >&2
  fi
else
  echo "WARNING: module command unavailable; using existing .venv python." >&2
fi

GPU_READY=true
if [[ "${HAVE_MODULE}" == "true" ]]; then
  if ! module load cuda/11.8.0; then
    echo "WARNING: could not load cuda/11.8.0; CPU fallback will be used." >&2
    GPU_READY=false
  fi
  if ! module load cudnn/8.6.0.163-11.8-gpu; then
    echo "WARNING: could not load cudnn/8.6.0.163-11.8-gpu; CPU fallback will be used." >&2
    GPU_READY=false
  fi
else
  echo "WARNING: module command unavailable; CPU fallback will be used." >&2
  GPU_READY=false
fi

source .venv/bin/activate

VENV_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "${VENV_VERSION}" != "3.10.3" ]]; then
  echo "ERROR: .venv must use Python 3.10.3 (found ${VENV_VERSION})." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

set -a
source .env
set +a

if [[ "${QWEN_MODEL_ID:-}" == "Qwen/Qwen3.5-8B" ]]; then
  echo "WARNING: QWEN_MODEL_ID=Qwen/Qwen3.5-8B is invalid; switching to Qwen/Qwen3.5-9B." >&2
  export QWEN_MODEL_ID="Qwen/Qwen3.5-9B"
fi

if [[ -z "${QWEN_MODEL_PATH:-}" ]]; then
  export QWEN_MODEL_PATH="./models/local_models/qwen3.5-9b"
fi

if [[ "${QWEN_MODEL_PATH:-}" == "./models/local_models/qwen3.5-8b" ]]; then
  export QWEN_MODEL_PATH="./models/local_models/qwen3.5-9b"
fi

if [[ "${GPU_READY}" != "true" ]]; then
  export QWEN_DEVICE=cpu
fi

python models/qwen_test.py --offline
