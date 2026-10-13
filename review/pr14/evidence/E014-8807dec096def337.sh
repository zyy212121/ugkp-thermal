#!/usr/bin/env bash
set -o pipefail
base=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/remaining
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
export CUDA_HOME=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/toolchains/cuda
export FOAM_USER_APPBIN="$base/bin" FOAM_USER_LIBBIN="$base/lib"
export TMPDIR="$base/tmp" UGKWP_CUDA_ARCH=sm_80 UGKWP_GAS_SPECIES=2
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$TMPDIR"
cd "$base/source"
app="$1"
case "$app" in
 gasUGKP|FSH) script="applications/$app/private_backend/build_private_backend.sh";;
 CHT) script="applications/CHT/tools/build_cuda_solver.sh";;
 *) exit 2;;
esac
mkdir -p "$base/$app"
(env | grep -E "^(WM_|FOAM_|CUDA_HOME=|UGKWP_|TMPDIR=|LD_LIBRARY_PATH=|PATH=|CXX=)" | sort; "$CUDA_HOME/bin/nvcc" --version; c++ --version; command -v wmake; echo "script=$script"; sha256sum "$script") > "$base/$app/toolchain.txt"
{ time -p bash -x "$script"; } > "$base/$app/build.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$base/$app/exit.txt"
tail -35 "$base/$app/build.log"
exit "$rc"
