#!/usr/bin/env bash
# Parent-declared checkpoint execution only. Never called by prepare.py.
source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export GAS_UGKP_CUDA_STACK_BYTES="${GAS_UGKP_CUDA_STACK_BYTES:-65536}"
export CHT_CUDA_STACK_BYTES="${CHT_CUDA_STACK_BYTES:-65536}"
export CHMT_CUDA_STACK_BYTES="${CHMT_CUDA_STACK_BYTES:-65536}"
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/run_batch.py" run "$1"
