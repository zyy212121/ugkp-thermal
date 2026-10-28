#!/usr/bin/env bash
source /opt/openfoam10/etc/bashrc
set -eo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
export FOAM_USER_APPBIN="$R/bin" GAS_UGKP_BUILD="$R/build-evidence/gas-probe" NVCC="$R/build-evidence/nvcc-bounded-probe"
export CUDA_HOME=/usr/local/cuda UGKWP_GAS_SPECIES=2 UGKWP_CUDA_ARCH=sm_89 WM_NCOMPPROCS=1
/usr/bin/time -v bash "$R/tool/gasUGKP/build_cuda_solver.sh" > "$R/build-evidence/gas-probe.log" 2>&1
