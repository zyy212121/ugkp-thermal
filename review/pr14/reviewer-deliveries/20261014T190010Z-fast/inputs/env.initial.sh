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
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
exec "$@"
