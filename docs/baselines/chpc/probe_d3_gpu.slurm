#!/bin/bash
#SBATCH --job-name=plcbench-d3-gpu-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/d3_gpu_probe_%j.out
#SBATCH --error=logs/d3_gpu_probe_%j.err

set -euo pipefail
: "${D3_ROOT:?}"
"$D3_ROOT/env/bin/python" - <<'PY'
import json
import subprocess

import torch

data = {
    'torch_version': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    'nvidia_smi': subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
        '--format=csv,noheader'], text=True).strip(),
}
print(json.dumps(data, sort_keys=True))
assert data['cuda_available'], 'PyTorch cannot initialize the allocated CUDA GPU'
assert data['gpu_memory_bytes'] >= 75_000 * 1024**2, 'allocated GPU is too small for D3'
PY
