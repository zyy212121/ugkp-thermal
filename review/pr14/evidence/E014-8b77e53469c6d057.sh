#!/usr/bin/env bash
# Standard native host compile/link only; reuse immutable unchanged CUDA archive.
set -o pipefail
base=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/mapper-access-final
saved_ld=${LD_LIBRARY_PATH:-}
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
set -e
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${saved_ld:+:$saved_ld}"
export CUDA_HOME=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/toolchains/cuda
export FOAM_USER_APPBIN="$base/bin" FOAM_USER_LIBBIN="$base/lib" TMPDIR="$base/tmp"
export UGKWP_CUDA_EXE_INC=-DUGKWP_USE_CUDA
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$TMPDIR"
cd "$base/source/applications/CHT"
export UGKWP_CUDA_EXE_LIBS="$PWD/gpu/libugkpcht_cuda_internal.a -L$CUDA_HOME/lib64 -lcudart"
(env | grep -E '^(WM_|FOAM_|CUDA_HOME=|UGKWP_|TMPDIR=|LD_LIBRARY_PATH=|PATH=|CXX=)' | sort; c++ --version; command -v wmake) > "$base/toolchain.txt"
{ time -p wmake; } > "$base/CHT-build.log" 2>&1
printf '0\n' > "$base/CHT-exit.txt"
# Never run the linked application.
