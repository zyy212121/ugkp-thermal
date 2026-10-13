#!/usr/bin/env bash
set -o pipefail
base=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/restart-codec-cxx14
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
set -e
export CUDA_HOME=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/toolchains/cuda
export FOAM_USER_APPBIN="$base/bin" FOAM_USER_LIBBIN="$base/lib" TMPDIR="$base/tmp"
export UGKWP_CUDA_EXE_INC=-DUGKWP_USE_CUDA
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$TMPDIR"
for app in CHT FSH; do
 cd "$base/source/applications/$app"
 if [ "$app" = CHT ]; then
  export UGKWP_CUDA_EXE_LIBS="$PWD/gpu/libugkpcht_cuda_internal.a -L$CUDA_HOME/lib64 -lcudart"
 else
  unset UGKWP_CUDA_EXE_LIBS
 fi
 wmake > "$base/$app-build.log" 2>&1
 echo 0 > "$base/$app-exit.txt"
done
