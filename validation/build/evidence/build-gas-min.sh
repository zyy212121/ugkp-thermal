#!/usr/bin/env bash
source /opt/openfoam10/etc/bashrc
set -eo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
# Diagnostic experiment only: keep an unsuccessful compiler below global OOM.
ulimit -v 10485760
export FOAM_USER_APPBIN="$R/bin" GAS_UGKP_BUILD="$R/build-evidence/gas-min"
export NVCC=/usr/local/cuda/bin/nvcc UGKWP_CUDA_FAST_COMPILE=min
export CUDA_HOME=/usr/local/cuda UGKWP_GAS_SPECIES=2 UGKWP_CUDA_ARCH=sm_89 WM_NCOMPPROCS=1
/usr/bin/time -v bash "$R/tool/gasUGKP/build_cuda_solver.sh" > "$R/build-evidence/gas-min.log" 2>&1
