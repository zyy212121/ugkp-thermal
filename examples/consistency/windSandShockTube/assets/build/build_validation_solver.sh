#!/usr/bin/env bash

source /opt/openfoam10/etc/bashrc
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unified_root="$(cd "${case_dir}/../../.." && pwd)"
canonical="${unified_root}/applications/gasUGKP"
asset="${case_dir}/assets/solver_source"
build_root="${case_dir}/.validation-build"
stage="${build_root}/source"
backend_source="${build_root}/backend-source"
backend_obj="${build_root}/backend"
bin_dir="${case_dir}/.validation-bin"

"${case_dir}/assets/build/clean_validation_solver.sh"
mkdir -p \
    "${stage}/gpu" "${stage}/Make" \
    "${backend_source}" "${backend_obj}" "${bin_dir}"

python3 "${asset}/tests/test_constant_response_time_contract.py"

install -m 0644 "${canonical}"/*.H "${stage}/"
install -m 0644 "${canonical}"/*.C "${stage}/"
cp -aL "${canonical}/gpu/." "${stage}/gpu/"
patch --batch --forward -d "${stage}/gpu" -p0 \
    < "${asset}/GpuResidentStrict.axial-fluctuation.patch"
install -m 0644 \
    "${asset}/gpu/GpuDragModel.H" \
    "${stage}/gpu/GpuDragModel.H"
install -m 0644 \
    "${unified_root}/common/GpuSchedulingConfiguration.H" \
    "${stage}/gpu/GpuSchedulingConfiguration.H"
install -m 0644 "${canonical}/Make/options" "${stage}/Make/options"
install -m 0644 "${canonical}/Make/files" "${stage}/Make/files"
sed -i \
    's#$(FOAM_USER_APPBIN)/gasUGKP#$(FOAM_USER_APPBIN)/windSandUGKP#' \
    "${stage}/Make/files"

cp -aL "${canonical}/private_backend/." "${backend_source}/"
install -m 0644 \
    "${asset}/private_backend/GpuDragModels.cuh" \
    "${backend_source}/GpuDragModels.cuh"
patch --batch --forward -d "${backend_source}" -p0 \
    < "${asset}/GpuResidentStrict.constant-response-time.patch"

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cuda_arch="${UGKWP_CUDA_ARCH:-sm_89}"
case "${GAS_UGKP_FMAD:-1}" in
    1|on|true|yes) fmad_flag="--fmad=true" ;;
    0|off|false|no) fmad_flag="--fmad=false" ;;
    *) echo "GAS_UGKP_FMAD must be 0 or 1" >&2; exit 2 ;;
esac

"${cuda_home}/bin/nvcc" \
    -std=c++17 -O3 "${fmad_flag}" -arch="${cuda_arch}" \
    -Xcompiler -fPIC \
    -I"${stage}/gpu" -I"${backend_source}" \
    -c "${backend_source}/GpuResidentStrict.cu" \
    -o "${backend_obj}/GpuResidentStrict.o"

ar rcs "${backend_obj}/libwindsand_cuda_backend.a" \
    "${backend_obj}/GpuResidentStrict.o"

c++ -std=c++17 -O3 \
    -I"${stage}/gpu" -I"${backend_source}" \
    "${backend_source}/GpuBackendServer.C" \
    "${backend_obj}/libwindsand_cuda_backend.a" \
    -L"${cuda_home}/lib64" -Wl,-rpath,"${cuda_home}/lib64" -lcudart \
    -o "${bin_dir}/windSandUGKPCudaBackend"

(
    cd "${stage}"
    FOAM_USER_APPBIN="${bin_dir}" \
        UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA" \
        wmake
)

test -x "${bin_dir}/windSandUGKP"
test -x "${bin_dir}/windSandUGKPCudaBackend"
if ldd "${bin_dir}/windSandUGKP" | grep -Eq 'libcuda|libcudart'; then
    echo "Validation frontend directly links CUDA" >&2
    exit 1
fi
if ldd "${bin_dir}/windSandUGKPCudaBackend" \
    | grep -Eq 'libOpenFOAM|libfiniteVolume|libmeshTools'; then
    echo "Validation backend directly links OpenFOAM" >&2
    exit 1
fi

printf 'canonical_source=%s\nadapter=%s\nfrontend=%s\nbackend=%s\n' \
    "$(find "${canonical}" -type f \
        \( -name '*.C' -o -name '*.H' -o -name '*.cu' -o -name '*.cuh' \) \
        -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')" \
    "$(sha256sum "${asset}/GpuResidentStrict.constant-response-time.patch" \
        "${asset}/GpuResidentStrict.axial-fluctuation.patch" \
        | sha256sum | awk '{print $1}')" \
    "$(sha256sum "${bin_dir}/windSandUGKP" | awk '{print $1}')" \
    "$(sha256sum "${bin_dir}/windSandUGKPCudaBackend" | awk '{print $1}')"
