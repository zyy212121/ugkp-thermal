#!/usr/bin/env bash
set -o pipefail
source /opt/openfoam10/etc/bashrc
set -e
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
review_root=$(git -C "$script_dir" rev-parse --show-toplevel)
export FOAM_USER_APPBIN="$review_root/work/install/ns2/bin"
export FOAM_USER_LIBBIN="$review_root/work/install/ns2/lib"
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$review_root/work/install/ns10/bin"
export PATH="$FOAM_USER_APPBIN:/usr/local/cuda/bin:$FOAM_APPBIN:$WM_PROJECT_DIR/wmake:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:/usr/local/cuda/lib64:/usr/lib/wsl/lib:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI"
export CUDA_HOME=/usr/local/cuda
export UGKWP_CUDA_ARCH=sm_89
export CHMT_CUDA_ARCH=sm_89
export UGKWP_GAS_SPECIES=2
export CHMT_BUILD="$review_root/work/build/CHMT2"
export UGKWP_CUDA_LOG_PHASE=pr14_reviewer_fast
export UGKP_DEVELOPMENT_PROBES=0
export GAS_UGKP_CUDA_BACKEND="$FOAM_USER_APPBIN/gasUGKPCudaBackend"
export FSH_CUDA_BACKEND="$FOAM_USER_APPBIN/FSHCudaBackend"
export WM_NCOMPPROCS=2
exec "$@"
