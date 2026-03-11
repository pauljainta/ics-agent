#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  if ! module load python/3.10.3; then
    echo "WARNING: could not load module python/3.10.3; using existing .venv python." >&2
  fi
else
  echo "WARNING: module command unavailable; using existing .venv python." >&2
fi

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

source .venv/bin/activate

VENV_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "${VENV_VERSION}" != "3.10.3" ]]; then
  echo "ERROR: .venv must use Python 3.10.3 (found ${VENV_VERSION})." >&2
  echo "Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.5-0.8B}"
MODEL_DIR="${QWEN_MODEL_PATH:-./models/local_models/qwen3.5-0.8b}"

mkdir -p "${MODEL_DIR}"

echo "Downloading model '${MODEL_ID}' into '${MODEL_DIR}'"
if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}" \
    --local-dir-use-symlinks False
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found in .venv. Run: bash scripts/setup_env.sh" >&2
  exit 1
fi

echo "Download complete."
echo "QWEN_MODEL_PATH=${MODEL_DIR}"
