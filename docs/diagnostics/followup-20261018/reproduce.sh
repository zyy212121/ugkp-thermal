#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${1:-"$ROOT/.review-followup-reproduction"}
source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
set -euo pipefail
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.1}
export NVCC="$CUDA_HOME/bin/nvcc" UGKWP_CUDA_ARCH=sm_89 CHMT_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2
export FOAM_USER_APPBIN="$OUT/bin" FOAM_USER_LIBBIN="$OUT/lib"
export PATH="$FOAM_USER_APPBIN:$CUDA_HOME/bin:$FOAM_APPBIN:/usr/bin:/bin:$WM_PROJECT_DIR/wmake"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$CUDA_HOME/lib64"
export CUDA_CACHE_PATH="$OUT/cuda-cache" XDG_CACHE_HOME="$OUT/cache" TMPDIR="$OUT/tmp"
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_CACHE_PATH" "$TMPDIR"
export PYTHONPATH="$ROOT:$ROOT/validation/tests/gas_transport:$ROOT/validation/tests/gas_wall"
cd "$ROOT"
python3 -m pytest -q validation/tests/gas_wall/test_wall_newton_globalization.py validation/tests/gas_transport/test_followup_fraction_and_cp.py validation/tests/test_chmt_audit_overflow_followup.py validation/tests/test_stencil_consumer_contracts.py
# Request GPU work explicitly, on an available GPU, using: reproduce.sh OUT gpu
if [[ ${2:-} == gpu ]]; then
 "$NVCC" -std=c++17 -O3 -lineinfo -arch=sm_89 --fmad=true -I"$ROOT" validation/tests/gas_wall/fixtures/sst_globalization_gpu.cu -o "$OUT/wall-gpu"
 "$OUT/wall-gpu"
 for tool in memcheck racecheck synccheck; do "$CUDA_HOME/bin/compute-sanitizer" --tool "$tool" --error-exitcode 3 "$OUT/wall-gpu"; done
fi
