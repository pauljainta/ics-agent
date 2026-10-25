#!/bin/bash
#SBATCH --job-name=plcbench-vllm
# CONFIRM YOUR ALLOCATION. A different project's School-of-Computing
# notchpeak account/partition; do NOT assume it carries to SPHERE/ics-agent.
# Run `myallocation` on CHPC and set the account/partition CHPC returns.
#SBATCH --account=REPLACE_ME_ACCOUNT
#SBATCH --partition=REPLACE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# 1x A100 is enough for Qwen2.5-32B-Instruct-AWQ (~20 GB). For bf16 32B use two
# 40 GB A100s (--gres=gpu:a100:2) and --tensor-parallel-size 2 below.
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/plcbench_vllm_%j.out
#SBATCH --error=logs/plcbench_vllm_%j.err

set -euo pipefail

# --- model + serving config (one frozen deployment for BOTH comparison arms) ---
MODEL="${PLCBENCH_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"   # AWQ fits 1x40GB A100
QUANT="${PLCBENCH_QUANT:-awq}"                              # drop for bf16 on 2 GPUs
TP="${PLCBENCH_TP:-1}"                                      # set 2 for bf16 32B on 2x A100
PORT="${PLCBENCH_PORT:-8000}"
BIND_HOST="${PLCBENCH_BIND_HOST:-127.0.0.1}"
MAX_LEN="${PLCBENCH_MAX_LEN:-8192}"
# Keep weights/cache on a confirmed scratch path. CHPC does not always set
# SCRATCH, so require an explicit HF_HOME rather than resolving to /hf.
: "${HF_HOME:?Set HF_HOME to a writable scratch cache before sbatch}"
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Read the key at job startup so it is not embedded in a Slurm script or
# printed with vLLM's command-line arguments.
if [ -n "${PLCBENCH_API_KEY_FILE:-}" ]; then
  export VLLM_API_KEY="$(<"$PLCBENCH_API_KEY_FILE")"
elif [ -n "${PLCBENCH_API_KEY:-}" ]; then
  export VLLM_API_KEY="$PLCBENCH_API_KEY"
fi

module load miniconda3/25.9.1
source activate "${PLCBENCH_VLLM_ENV:-vllm}" || conda activate "${PLCBENCH_VLLM_ENV:-vllm}"

echo "serving $MODEL on $(hostname) port $PORT (tp=$TP quant=${QUANT:-none})"

ARGS=(--port "$PORT" --host "$BIND_HOST"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN"
      --gpu-memory-utilization 0.92 --served-model-name plcbench)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")

# `vllm serve` (newer) or the module entrypoint (older) both expose /v1.
if command -v vllm >/dev/null 2>&1; then
  exec vllm serve "$MODEL" "${ARGS[@]}"
else
  exec python -m vllm.entrypoints.openai.api_server --model "$MODEL" "${ARGS[@]}"
fi
