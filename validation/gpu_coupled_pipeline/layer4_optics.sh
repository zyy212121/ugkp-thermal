#!/usr/bin/env bash
# Source OpenFOAM 10 first; use the preserved 88bc checkout as the oracle.
set -euo pipefail
if [[ $# != 3 ]]; then echo "usage: $0 REPOSITORY FROZEN_BASELINE OUTPUT" >&2; exit 2; fi
root=$(realpath "$1"); baseline=$(realpath "$2"); out=$(realpath -m "$3")
mkdir "$out"
cuda=${CUDA_HOME:-/usr/local/cuda}
python3 - "$root" "$baseline" "$out" <<'PY'
from pathlib import Path
import sys
root,baseline,out=map(Path,sys.argv[1:])
for ext in ['H','C']:
    text=(baseline/f'applications/CHT/thermal/MieTable.{ext}').read_text().replace('namespace gpuThermal','namespace originalGpuThermal')
    text=text.replace('GPU_THERMAL_MIE_TABLE_H','ORIGINAL_GPU_THERMAL_MIE_TABLE_H').replace('#include "MieTable.H"','#include "OriginalMieTable.H"')
    (out/f'OriginalMieTable.{ext}').write_text(text)
text=(baseline/'applications/CHT/thermal/MieDoRadiationMath.H').read_text().replace('namespace gpuThermal','namespace originalGpuThermal')
text=text.replace('GPU_THERMAL_MIE_DO_RADIATION_MATH_H','ORIGINAL_GPU_THERMAL_MIE_DO_RADIATION_MATH_H')
(out/'OriginalMieDoRadiationMath.H').write_text(text)
for kind in ['ordinary','zero_scattering','zero_diameter','negative_diameter','equal_logarithms','wide_temperature']:
    temperatures=[300.,700.,1500.,3000.];diameters=[1e-5,2e-5,.0002];mu=[-1.,-.5,0.,.3,1.]
    if kind=='zero_diameter':diameters=[0.,2e-5,.0002]
    if kind=='negative_diameter':diameters=[-2e-4,-2e-5,-1e-5]
    if kind=='equal_logarithms':diameters=[1e100,1.0000000000000002e100,1e101]
    if kind=='wide_temperature':temperatures=[-1e308,1e308,1.1e308,1.2e308]
    lines=['Tvals '+' '.join(map(repr,temperatures)),'Dvals '+' '.join(map(repr,diameters)),
        'Muvals '+' '.join(map(repr,mu)),'PlanckBandFractions 0.1 0.3 0.7 0.9','DATA']
    for d in range(3):
        for t in range(4):
            qa=.2+.03*d+.01*t;qs=0. if kind=='zero_scattering' else .8+.02*d+.04*t;g=.1+.03*d+.02*t
            lines.append(f'NODE {d} {t} {qa!r} {qs!r} {qa+qs!r}')
            lines.append('PHI '+' '.join(repr((1-g*g)/(1+g*g-2*g*m)**1.5) for m in mu))
    (out/(kind+'.table')).write_text('\n'.join(lines+['END_MIE_TABLE','']))
PY
"$cuda/bin/nvcc" -std=c++17 -O2 --fmad=false -arch="${UGKWP_CUDA_ARCH:-sm_89}" -c "$root/common/gpu/MieOpticalProperties.cu" -o "$out/MieOpticalProperties.o" > "$out/build-cuda.log" 2>&1
thermal=$root/applications/CHT/thermal
g++ -std=c++17 -O2 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \
 -I"$root/common" -I"$thermal" -I"$out" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" \
 -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" "$root/validation/gpu_coupled_pipeline/mie_optics_cuda.cpp" \
 "$thermal/MieTable.C" "$out/OriginalMieTable.C" "$out/MieOpticalProperties.o" \
 -L"$FOAM_LIBBIN" -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream \
 -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart -o "$out/optics-test" > "$out/build-test.log" 2>&1
for table in "$out"/*.table; do
 "$out/optics-test" "$table" > "$table.log" 2>&1
done
"$cuda/bin/compute-sanitizer" --tool memcheck --leak-check full --error-exitcode 99 "$out/optics-test" "$out/ordinary.table" > "$out/memcheck.log" 2>&1
sha256sum "$out/optics-test" "$out/MieOpticalProperties.o" "$out"/OriginalMieTable.* "$out/OriginalMieDoRadiationMath.H" > "$out/binaries.sha256"
printf '%s\n' 'PASS frozen Mie table optics, IEEE edge cases and CUDA memcheck'
