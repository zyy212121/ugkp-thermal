#!/usr/bin/env bash
# Build all applications with checkout-local object and executable paths.
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
if [[ -z ${WM_PROJECT_DIR:-} ]]; then
    source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
fi
set -euo pipefail
export FOAM_USER_APPBIN="$source_root/bin"
export FOAM_USER_LIBBIN="$source_root/lib"
export GAS_UGKP_BUILD="$source_root/applications/gasUGKP/private_backend/build/$WM_OPTIONS"
export GAS_UGKP_STEADY_BUILD="$source_root/applications/gasUGKPSteady/private_backend/build/$WM_OPTIONS"
export CHMT_BUILD="$source_root/applications/CHMT/.build/native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NVCC="$CUDA_HOME/bin/nvcc"
export WM_NCOMPPROCS="${WM_NCOMPPROCS:-1}"
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
exec bash "$source_root/Allwmake"
