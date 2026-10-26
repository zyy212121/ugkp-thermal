#!/usr/bin/env bash
# Source OpenFOAM 10 first. TABLE_DIRECTORY contains the six layer4_optics.sh tables.
set -euo pipefail
if [[ $# != 4 ]]; then echo "usage: $0 REPOSITORY FROZEN_BASELINE TABLE_DIRECTORY OUTPUT" >&2; exit 2; fi
root=$(realpath "$1"); baseline=$(realpath "$2"); tables=$(realpath "$3"); out=$(realpath -m "$4")
mkdir "$out"
cuda=${CUDA_HOME:-/usr/local/cuda}
python3 - "$baseline" "$out" <<'PY'
from pathlib import Path
import sys
baseline,out=map(Path,sys.argv[1:])
for name in ['MieTable.H','MieTable.C','MieDoRadiationMath.H','ParticleRadiationImplicitMath.H']:
    text=(baseline/'applications/CHT/thermal'/name).read_text().replace('namespace gpuThermal','namespace originalGpuThermal')
    text=text.replace('#include "MieTable.H"','#include "OriginalMieTable.H"')
    for guard in ['GPU_THERMAL_MIE_TABLE_H','GPU_THERMAL_MIE_DO_RADIATION_MATH_H','GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H']:
        text=text.replace(guard,'ORIGINAL_'+guard)
    (out/('Original'+name)).write_text(text)
PY
for unit in MieOpticalProperties ParticleRadiationMaterial; do
 "$cuda/bin/nvcc" -std=c++17 -O2 --fmad=false -arch="${UGKWP_CUDA_ARCH:-sm_89}" -c "$root/common/gpu/$unit.cu" -o "$out/$unit.o" > "$out/build-$unit.log" 2>&1
done
thermal=$root/applications/CHT/thermal
g++ -std=c++17 -O2 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \
 -I"$root/common" -I"$thermal" -I"$out" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$root/validation/gpu_coupled_pipeline/particle_radiation_cuda.cpp" "$thermal/MieTable.C" "$out/OriginalMieTable.C" "$out/MieOpticalProperties.o" "$out/ParticleRadiationMaterial.o" \
 -L"$FOAM_LIBBIN" -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart \
 -o "$out/material-test" > "$out/build-test.log" 2>&1
for kind in ordinary zero_scattering zero_diameter negative_diameter equal_logarithms wide_temperature; do
 "$out/material-test" "$tables/$kind.table" > "$out/$kind.log" 2>&1
done
"$cuda/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 "$out/material-test" "$tables/ordinary.table" > "$out/memcheck.log" 2>&1
sha256sum "$out/material-test" "$out"/*.o "$out"/Original* > "$out/binaries.sha256"
printf '%s\n' 'PASS frozen implicit particle roots, conservative boundary allocation and CUDA memcheck'
