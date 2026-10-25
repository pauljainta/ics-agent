#!/bin/bash
#SBATCH --job-name=plcbench-d3-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d3_env_%j.out
#SBATCH --error=logs/d3_env_%j.err

set -euo pipefail
: "${D3_ROOT:?}"
export D3_PHASE=env
bash -l "$D3_ROOT/prepare_d3_systemone.sh"
