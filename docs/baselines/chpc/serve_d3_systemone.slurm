#!/bin/bash
#SBATCH --job-name=plcbench-d3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/d3_%j.out
#SBATCH --error=logs/d3_%j.err

set -euo pipefail
set +x

# Supply account, partition, and a single >=80 GB GPU with sbatch arguments.
# D3_ROOT points to the scratch directory prepared by prepare_d3_systemone.sh.
: "${D3_ROOT:?D3_ROOT must name the prepared scratch directory}"
: "${D3_API_KEY_FILE:?D3_API_KEY_FILE must name a private key file}"
: "${D3_LOGIN_IP:?D3_LOGIN_IP must be the current login-node source IP}"
REVISION=cf15549a78fc52598ddd1951f321c58372bbd64f
MODEL="$D3_ROOT/hf/hub/models--vllm-sr--d3/snapshots/$REVISION"
PYTHON="$D3_ROOT/env/bin/python"

test -s "$MODEL/MODEL_MANIFEST.json"
test -s "$MODEL/d3_server.py"
test -x "$PYTHON"
test -s "$D3_API_KEY_FILE"
test -s "$D3_ROOT/plcbench_systemone_relay.py"
test "$(wc -c < "$D3_API_KEY_FILE")" -ge 32

D3_RELAY_BIND_HOST=$(ip -4 route get "$D3_LOGIN_IP" | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}')
test -n "$D3_RELAY_BIND_HOST"
export D3_RELAY_BIND_HOST

GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
test "$GPU_MEMORY_MIB" -ge 75000 || {
  echo "D3 requires one GPU with at least 75000 MiB; got ${GPU_MEMORY_MIB} MiB" >&2
  exit 1
}

export HF_HOME="$D3_ROOT/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DECISION_API_KEY="$(< "$D3_API_KEY_FILE")"
mkdir -p "$D3_ROOT/runs"
"$PYTHON" - "$MODEL" "$D3_ROOT/runs/d3_${SLURM_JOB_ID}_identity.json" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import torch
import transformers

model, destination = map(pathlib.Path, sys.argv[1:])
assert torch.cuda.is_available(), 'CUDA is unavailable'
data = {
    'schema_version': 'plcbench.d3-deployment/v1',
    'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'model_id': 'd3',
    'model_source': 'vllm-sr/d3',
    'weight_revision': model.name,
    'server_sha256': hashlib.sha256((model / 'd3_server.py').read_bytes()).hexdigest(),
    'manifest_sha256': hashlib.sha256((model / 'MODEL_MANIFEST.json').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
    'job_script_sha256': hashlib.sha256(
        (model.parents[4] / 'serve_d3_systemone.slurm').read_bytes()).hexdigest(),
    'pip_freeze_sha256': hashlib.sha256(
        (model.parents[4] / 'runs' / f'd3_{model.name}_pip_freeze.txt').read_bytes()).hexdigest(),
    'relay_sha256': hashlib.sha256(
        (model.parents[4] / 'plcbench_systemone_relay.py').read_bytes()).hexdigest(),
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
    'cuda_runtime': torch.version.cuda,
    'driver_version': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        text=True).splitlines()[0].strip(),
    'gpu_name': torch.cuda.get_device_name(0),
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'slurm_job_id': os.environ['SLURM_JOB_ID'],
    'node': os.environ['SLURMD_NODENAME'],
    'bind_host': '127.0.0.1',
    'port': 18080,
    'relay_bind_host': os.environ['D3_RELAY_BIND_HOST'],
    'relay_port': 18081,
    'relay_allow_ip': os.environ['D3_LOGIN_IP'],
}
destination.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
PY

"$PYTHON" "$MODEL/d3_server.py" --model "$MODEL" --verify full \
  --device cuda:0 --name d3 --host 127.0.0.1 --port 18080 &
D3_PID=$!
"$PYTHON" "$D3_ROOT/plcbench_systemone_relay.py" \
  --key-file "$D3_API_KEY_FILE" --allow-ip "$D3_LOGIN_IP" \
  --listen-host "$D3_RELAY_BIND_HOST" --listen-port 18081 &
RELAY_PID=$!
trap 'kill "$D3_PID" "$RELAY_PID" 2>/dev/null || true' EXIT TERM INT
wait -n "$D3_PID" "$RELAY_PID"
