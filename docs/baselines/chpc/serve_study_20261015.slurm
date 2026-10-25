#!/bin/bash
#SBATCH --job-name=plcbench-study-qwen
# Supply the confirmed account and partition with sbatch, not in this script.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/study_%j.out
#SBATCH --error=logs/study_%j.err

set -euo pipefail
set +x
module load vllm/current
BASE="${PLCBENCH_STUDY_ROOT:?set the private scratch directory at submission}"
LOGIN_IP="${PLCBENCH_LOGIN_IP:?set the current login-node source IP at submission}"
REVISION=5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c
MODEL_DIR="$BASE/hf/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/$REVISION"
SIF=/opt/research-runtime/vllm_current.sif
EXPECTED_SIF=551a6c4c886cfe8a8b6c24be9bf47744e9462d70884e8bd7ba88474fa8375f09
EXPECTED_RELAY=4a67ff9bfa32415d18239304b9fedc34a31a2fa7cc877b15e05d03abdda08201
test -s "$MODEL_DIR/config.json"
test -s "$BASE/api_key_study"
test "$(sha256sum "$SIF" | cut -d' ' -f1)" = "$EXPECTED_SIF"
test "$(sha256sum "$BASE/plcbench_auth_relay.py" | cut -d' ' -f1)" = "$EXPECTED_RELAY"
export HF_HOME="$BASE/hf" HF_HUB_OFFLINE=1
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_VLLM_API_KEY="$(<"$BASE/api_key_study")"
python3 "$BASE/plcbench_auth_relay.py" \
  --key-file "$BASE/api_key_study" --allow-ip "$LOGIN_IP" \
  --listen-port 8001 &
RELAY_PID=$!
apptainer run --fakeroot --nv "$SIF" "$MODEL_DIR" \
  --served-model-name plcbench --host 127.0.0.1 --port 8000 \
  --quantization awq --tensor-parallel-size 1 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill "$RELAY_PID" "$VLLM_PID" 2>/dev/null || true' EXIT TERM INT
wait -n "$RELAY_PID" "$VLLM_PID"
