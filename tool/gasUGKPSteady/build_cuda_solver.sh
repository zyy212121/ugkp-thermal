#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
gas_species="${UGKWP_GAS_SPECIES-2}"
build="${GAS_UGKP_STEADY_BUILD:-$source_root/applications/gasUGKPSteady/private_backend/build/$WM_OPTIONS}"
case "$build" in /*) ;; *) build="$PWD/$build";; esac
mkdir -p "$build" "$FOAM_USER_APPBIN"
"${NVCC:-$cuda_home/bin/nvcc}" -std=c++17 --expt-relaxed-constexpr \
 -DUGKWP_GAS_SPECIES="$gas_species" -O3 --fmad=true \
 -arch="${UGKWP_CUDA_ARCH:-sm_89}" -Xcompiler=-fPIC \
 -I"$source_root/applications/gasUGKP/gpu" -I"$source_root/common" \
 -c "$source_root/applications/gasUGKPSteady/private_backend/GpuSteadyBackend.cu" \
 -o "$build/GpuResidentStrict.o"
ar rcs "$build/libugkwp_steady_backend.a" "$build/GpuResidentStrict.o"
cd "$source_root/applications/gasUGKPSteady"
rm -f "Make/$WM_OPTIONS/gasUGKPSteady.o" "Make/$WM_OPTIONS/gasUGKPSteady.C.dep"
UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=$gas_species" \
UGKWP_CUDA_EXE_LIBS="$build/libugkwp_steady_backend.a -L$cuda_home/lib64 -Wl,-rpath,$cuda_home/lib64 -lcudart" wmake
