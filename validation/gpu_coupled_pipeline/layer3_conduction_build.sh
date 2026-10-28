#!/usr/bin/env bash
# Source OpenFOAM 10 first. The archive must come from the candidate CHT build.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 REPOSITORY FROZEN_BASELINE CANDIDATE_CUDA_ARCHIVE OUTPUT" >&2
    exit 2
fi
root=$(realpath "$1"); baseline=$(realpath "$2"); archive=$(realpath "$3"); out=$(realpath -m "$4")
mkdir "$out"
cuda=${CUDA_HOME:-/usr/local/cuda}
flags=(-std=c++17 -O2 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32
    -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude"
    -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude")
libs=(-L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream)
for kind in baseline candidate; do
    tree=$root; extra=()
    if [[ $kind == baseline ]]; then
        tree=$baseline
    else
        extra=("$archive" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart)
    fi
    thermal=$tree/applications/CHT/thermal
    g++ "${flags[@]}" -I"$thermal" "$root/validation/gpu_coupled_pipeline/solid_conduction.cpp" \
        "$thermal/GpuSolidThermalCoupler.C" "$thermal/GpuSolidThermalProperties.C" \
        "$thermal/GpuThermalExchangeState.C" "${extra[@]}" "${libs[@]}" \
        -o "$out/$kind" > "$out/build-$kind.log" 2>&1
    sha256sum "$thermal/GpuSolidThermalCoupler.C" "$thermal/GpuSolidThermalProperties.C" \
        "$thermal/GpuThermalExchangeState.C" > "$out/sources-$kind.sha256"
done
sha256sum "$archive" "$root/validation/gpu_coupled_pipeline/solid_conduction.cpp" \
    "$out/baseline" "$out/candidate" > "$out/binaries.sha256"
