#!/bin/bash
# Prepare the pinned official D3 package on CHPC scratch in two Slurm phases.
# D3_PHASE=env uses a CPU compute node; D3_PHASE=weights uses a data transfer node.

set -euo pipefail
set +x
umask 077
: "${D3_ROOT:?set D3_ROOT to a CHPC scratch directory}"
: "${D3_PHASE:?set D3_PHASE to env or weights}"
case "$D3_ROOT" in
  /opt/research-runtime/*) ;;
  *) echo 'D3_ROOT must be under CHPC scratch' >&2; exit 2 ;;
esac

REVISION=cf15549a78fc52598ddd1951f321c58372bbd64f
mkdir -p "$D3_ROOT/private" "$D3_ROOT/hf" "$D3_ROOT/runs"
case "$D3_PHASE" in
  env)
    type module >/dev/null 2>&1 || {
      echo 'CHPC module command is unavailable' >&2
      exit 1
    }
    module load miniconda3/25.9.1
    if [ ! -x "$D3_ROOT/env/bin/python" ]; then
      conda create -y -p "$D3_ROOT/env" python=3.12
    fi
    PYTHON="$D3_ROOT/env/bin/python"
    "$PYTHON" -m pip install --upgrade pip
    # The reference PyTorch kernels are sufficient for a first correctness trial.
    # The optional flash-linear-attention package is intentionally absent.
    "$PYTHON" -m pip install 'torch>=2.8' torchvision 'transformers==5.17.0' \
      'accelerate>=1.10' 'safetensors>=0.6' 'huggingface_hub>=1.0' \
      einops 'pillow>=10' opencv-python-headless
    "$PYTHON" -m pip freeze > "$D3_ROOT/runs/d3_${REVISION}_pip_freeze.txt"
    KEY_FILE="$D3_ROOT/private/api_key"
    if [ ! -e "$KEY_FILE" ]; then
      "$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(48))' > "$KEY_FILE"
    fi
    chmod 600 "$KEY_FILE"
    echo "Python environment: $D3_ROOT/env"
    echo "Private API key path: $KEY_FILE"
    ;;
  weights)
    AVAILABLE_BYTES=$(df -B1 --output=avail "$D3_ROOT" | tail -1 | tr -d ' ')
    test "$AVAILABLE_BYTES" -ge 70000000000 || {
      echo 'At least 70 GB free scratch space is required before downloading D3' >&2
      exit 1
    }
    test -x "$D3_ROOT/env/bin/hf"
    export HF_HOME="$D3_ROOT/hf"
    "$D3_ROOT/env/bin/hf" download vllm-sr/d3 --revision "$REVISION"
    MODEL="$D3_ROOT/hf/hub/models--vllm-sr--d3/snapshots/$REVISION"
    test -s "$MODEL/MODEL_MANIFEST.json"
    test -s "$MODEL/d3_server.py"
    echo "D3 snapshot prepared: $MODEL"
    ;;
  *) echo 'D3_PHASE must be env or weights' >&2; exit 2 ;;
esac
