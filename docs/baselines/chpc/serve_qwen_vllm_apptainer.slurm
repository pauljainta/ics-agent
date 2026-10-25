#!/bin/bash
#SBATCH --job-name=plcbench-vllm
#SBATCH --account=REPLACE_ME_ACCOUNT
#SBATCH --partition=REPLACE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/plcbench_vllm_%j.out
#SBATCH --error=logs/plcbench_vllm_%j.err

set -euo pipefail
set +x

# Submit from a directory with logs/ already created. Confirm account/partition
# with myallocation and replace the SBATCH placeholders before submission.
: "${HF_HOME:?Set a writable scratch Hugging Face cache}"
: "${PLCBENCH_MODEL_DIR:?Set the exact cached model snapshot directory}"
: "${PLCBENCH_WEIGHT_REVISION:?Set the exact 40-character weight commit}"
: "${PLCBENCH_SIF:?Set the resolved vLLM Apptainer image path}"
: "${PLCBENCH_SIF_SHA256:?Set its expected SHA-256}"
: "${PLCBENCH_API_KEY_FILE:?Set a private API key file path}"
test -s "$PLCBENCH_MODEL_DIR/config.json"
test -s "$PLCBENCH_API_KEY_FILE"
case "$PLCBENCH_MODEL_DIR" in
  */snapshots/"$PLCBENCH_WEIGHT_REVISION") ;;
  *) echo "model directory does not match the frozen revision" >&2; exit 1 ;;
esac
test "$(sha256sum "$PLCBENCH_SIF" | cut -d' ' -f1)" = "$PLCBENCH_SIF_SHA256"

export HF_HOME HF_HUB_OFFLINE=1
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
# vLLM logs CLI arguments. Pass the key only through the container environment.
export APPTAINERENV_VLLM_API_KEY="$(<"$PLCBENCH_API_KEY_FILE")"

module load vllm/current
exec apptainer run --fakeroot --nv "$PLCBENCH_SIF" "$PLCBENCH_MODEL_DIR" \
  --served-model-name plcbench \
  --host "${PLCBENCH_BIND_HOST:-127.0.0.1}" \
  --port "${PLCBENCH_PORT:-8000}" \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
