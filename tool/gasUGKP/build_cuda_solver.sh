#!/usr/bin/env bash
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail
solver_root="$source_root/applications/gasUGKP"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cuda_arch="${UGKWP_CUDA_ARCH:-sm_89}"
gas_species="${UGKWP_GAS_SPECIES-2}"
build="${GAS_UGKP_BUILD:-$solver_root/private_backend/build/$WM_OPTIONS}"
mkdir -p "$build" "$FOAM_USER_APPBIN"
probe_flags=()
case "${UGKP_DEVELOPMENT_PROBES:-0}" in
  1|on|true|yes) probe_flags=(-DUGKP_DEVELOPMENT_PROBES=1);;
  0|off|false|no) ;;
  *) echo 'ERROR: UGKP_DEVELOPMENT_PROBES must be 0 or 1' >&2; exit 2;;
esac
case "${GAS_UGKP_FMAD:-1}" in
  1|on|true|yes) fmad_flag=--fmad=true;;
  0|off|false|no) fmad_flag=--fmad=false;;
  *) echo 'ERROR: GAS_UGKP_FMAD must be 0 or 1' >&2; exit 2;;
esac
"${NVCC:-$cuda_home/bin/nvcc}" -std=c++17 --expt-relaxed-constexpr \
 -DUGKWP_GAS_SPECIES="$gas_species" -O3 "${probe_flags[@]}" "$fmad_flag" \
 -arch="$cuda_arch" -Xcompiler=-fPIC -I"$solver_root/gpu" -I"$source_root/common" \
 -c "$solver_root/private_backend/GpuResidentStrict.cu" -o "$build/GpuResidentStrict.o"
ar rcs "$build/libugkwp_cuda_backend.a" "$build/GpuResidentStrict.o"
cd "$solver_root"
# Rebuild the frontend for the selected species count and typed backend ABI.
rm -f "Make/$WM_OPTIONS/diluteUgkwpFoam.o" "Make/$WM_OPTIONS/diluteUgkwpFoam.C.dep"
UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=$gas_species" \
UGKWP_CUDA_EXE_LIBS="$build/libugkwp_cuda_backend.a -L$cuda_home/lib64 -Wl,-rpath,$cuda_home/lib64 -lcudart" wmake
test -x "$FOAM_USER_APPBIN/gasUGKP"
echo "Built resident single-process solver: $FOAM_USER_APPBIN/gasUGKP"
