#!/bin/bash
#SBATCH --job-name=plcbench-d3-weights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/d3_weights_%j.out
#SBATCH --error=logs/d3_weights_%j.err

set -euo pipefail
: "${D3_ROOT:?}"
export D3_PHASE=weights
bash -l "$D3_ROOT/prepare_d3_systemone.sh"
