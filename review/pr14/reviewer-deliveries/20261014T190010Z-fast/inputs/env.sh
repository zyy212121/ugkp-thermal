#!/usr/bin/env bash
set -o pipefail
source /opt/openfoam10/etc/bashrc
set -e
ROOT="$(git -C "$(dirname "$0")/../../../.." rev-parse --show-toplevel)"
export CUDA_HOME=/usr/local/cuda
export CUDA_ARCH=sm_89
export PATH="$CUDA_HOME/bin:$PATH"
export FOAM_USER_APPBIN="$ROOT/work/install/ns2/bin"
export FOAM_USER_LIBBIN="$ROOT/work/install/ns2/lib"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=2
export PATH="$FOAM_USER_APPBIN:$CUDA_HOME/bin:$FOAM_APPBIN:$WM_PROJECT_DIR/wmake:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$CUDA_HOME/lib64:/usr/lib/wsl/lib:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI"
export GAS_UGKP_CUDA_BACKEND="$FOAM_USER_APPBIN/gasUGKPCudaBackend"
export FSH_CUDA_BACKEND="$FOAM_USER_APPBIN/FSHCudaBackend"
export WM_NCOMPPROCS=2
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
exec "$@"
