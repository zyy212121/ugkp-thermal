#!/usr/bin/env bash

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
solver_root="${source_root}/applications/FSH"
private_root="${solver_root}/private_backend"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail
cd "${solver_root}"

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
source "$source_root/tool/cuda_build_flags.sh"
cuda_arch="${UGKWP_CUDA_ARCH:-sm_89}"
log_phase="${UGKWP_CUDA_LOG_PHASE:-fsh_finite_contact}"
development_probes="${UGKP_DEVELOPMENT_PROBES:-0}"
fmad_mode="${FSH_FMAD:-1}"
log_dir="${private_root}/build_logs/${log_phase}"
obj_dir="${private_root}/build/${WM_OPTIONS}"
backend_lib="${private_root}/libugkwp_cuda_backend.a"
backend_bin="${FOAM_USER_APPBIN}/FSHCudaBackend"
frontend_bin="${FOAM_USER_APPBIN}/FSH"

probe_nvcc_flags=()
case "${development_probes}" in
    1|on|true|yes)
        development_probes=1
        probe_nvcc_flags+=("-DUGKP_DEVELOPMENT_PROBES=1")
        ;;
    0|off|false|no)
        development_probes=0
        ;;
    *)
        echo "ERROR: UGKP_DEVELOPMENT_PROBES must be 0 or 1" >&2
        exit 2
        ;;
esac

case "${fmad_mode}" in
    1|on|true|yes)
        fmad_mode=1
        fmad_flag="--fmad=true"
        ;;
    0|off|false|no)
        fmad_mode=0
        fmad_flag="--fmad=false"
        ;;
    *)
        echo "ERROR: FSH_FMAD must be 0 or 1" >&2
        exit 2
        ;;
esac

mkdir -p "${log_dir}" "${obj_dir}"
log="${log_dir}/separated_build_$(date +%Y%m%d_%H%M%S).log"

{
    echo "[separated-build] solver_root=${solver_root}"
    echo "[separated-build] cuda_home=${cuda_home}"
    echo "[separated-build] cuda_arch=${cuda_arch}"
    echo "[separated-build] development_probes=${development_probes}"
    echo "[separated-build] fmad=${fmad_mode}"
    echo "[separated-build] WM_OPTIONS=${WM_OPTIONS}"

    rm -f "${obj_dir}"/*.o "${backend_lib}"

    "${NVCC:-$cuda_home/bin/nvcc}" "${ugkp_cuda_flags[@]}" \
        -std=c++17 \
        -O3 \
        "${probe_nvcc_flags[@]}" \
        "${fmad_flag}" \
        -arch="${cuda_arch}" \
        -Xcompiler -fPIC \
        -I"${solver_root}/gpu" -I"${solver_root}/../../common" \
        -c "${private_root}/GpuResidentStrict.cu" \
        -o "${obj_dir}/GpuResidentStrict.o"

    ar rcs "${backend_lib}" "${obj_dir}/GpuResidentStrict.o"

    c++ -std=c++17 -O3 \
        -I"${solver_root}/gpu" -I"${solver_root}/../../common" \
        "${private_root}/GpuBackendServer.C" \
        "${backend_lib}" \
        -L"${cuda_home}/lib64" -Wl,-rpath,"${cuda_home}/lib64" -lcudart \
        -o "${backend_bin}"

    rm -f \
        "${solver_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
        "${solver_root}/Make/${WM_OPTIONS}/gpu/GpuBackendClient.o" \
        "${solver_root}/Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
        "${solver_root}/Make/${WM_OPTIONS}/gpu/GpuBackendClient.C.dep"

    UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA" wmake

    test -x "${backend_bin}"
    test -x "${frontend_bin}"
    echo "[separated-build] frontend=${frontend_bin}"
    echo "[separated-build] backend=${backend_bin}"
} 2>&1 | tee "${log}"

echo "${log}"
