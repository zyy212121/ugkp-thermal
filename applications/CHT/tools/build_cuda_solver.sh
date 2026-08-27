#!/usr/bin/env bash

solver_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail
cd "${solver_root}"

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cuda_arch="${UGKWP_CUDA_ARCH:-sm_89}"
log_phase="${UGKWP_CUDA_LOG_PHASE:-ugkpcht_upgrade}"
log_dir="${solver_root}/docs/build_logs/${log_phase}"
obj_dir="${solver_root}/Make/${WM_OPTIONS}/gpu"
lib_path="${solver_root}/gpu/libugkpcht_cuda_internal.a"
cuda_bin="${FOAM_USER_APPBIN}/CHT"

mkdir -p "${log_dir}" "${obj_dir}"
log="${log_dir}/cuda_solver_build_$(date +%Y%m%d_%H%M%S).log"

{
    echo "[cuda-build] solver_root=${solver_root}"
    echo "[cuda-build] cuda_home=${cuda_home}"
    echo "[cuda-build] cuda_arch=${cuda_arch}"
    echo "[cuda-build] WM_OPTIONS=${WM_OPTIONS}"

    rm -f "${obj_dir}"/*.o "${lib_path}"

    "${cuda_home}/bin/nvcc" \
        -std=c++17 \
        -O3 \
        --fmad=false \
        -arch="${cuda_arch}" \
        -Xcompiler -fPIC \
        -I"${solver_root}/thermal" \
        -c "${solver_root}/gpu/GpuResidentStrict.cu" \
        -o "${obj_dir}/GpuResidentStrict.o"

    ar rcs \
        "${lib_path}" \
        "${obj_dir}/GpuResidentStrict.o"

    rm -f \
        "${solver_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
        "${solver_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
        "${solver_root}/Make/${WM_OPTIONS}/thermal/"*.o \
        "${solver_root}/Make/${WM_OPTIONS}/thermal/"*.dep

    UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA" \
    UGKWP_CUDA_EXE_LIBS="${lib_path} -L${cuda_home}/lib64 -lcudart" \
        wmake

    test -x "${cuda_bin}"
    echo "[cuda-build] cuda binary=${cuda_bin}"
} 2>&1 | tee "${log}"

echo "${log}"
