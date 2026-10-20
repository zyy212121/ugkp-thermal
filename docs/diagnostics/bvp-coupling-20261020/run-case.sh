#!/usr/bin/env bash
source /opt/openfoam10/etc/bashrc
set -eo pipefail
task_output=$(realpath "$1")
case_path=$(realpath "$2")
export FOAM_USER_APPBIN="$task_output/bin"
export FOAM_USER_LIBBIN="$task_output/lib"
export GAS_UGKP_CUDA_BACKEND="$FOAM_USER_APPBIN/gasUGKPCudaBackend"
export CUDA_CACHE_PATH="$task_output/cuda-cache"
export TMPDIR="$task_output/tmp"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN:/usr/local/cuda-13.1/lib64:${FOAM_EXT_LIBBIN:-}:/usr/lib/x86_64-linux-gnu/openmpi/lib"
exec "$FOAM_USER_APPBIN/gasUGKP" -case "$case_path"
