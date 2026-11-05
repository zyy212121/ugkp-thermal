#!/usr/bin/env bash
# Build the frozen or candidate production owner with test-only intensity export.
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "usage: $0 REPOSITORY OWNER_SOURCE OUTPUT [CUDA_ARCHIVE]" >&2; exit 2; fi
root=$(realpath "$1"); source=$(realpath "$2"); out=$(realpath -m "$3"); mkdir "$out"
thermal=$source/applications/CHT/thermal
python3 - "$thermal" "$out" <<'PY'
from pathlib import Path
import sys
thermal,out=map(Path,sys.argv[1:])
text=(thermal/'MieDoRadiationSolver.C').read_text()
needle='    MieDoResult result;'
assert text.count(needle)==1
export='''    static int testExport=0;
    std::ofstream angular((mesh.time().path()/("radiation-directions-"+std::to_string(testExport++)+".bin")).c_str(),std::ios::binary);
    forAll(intensities,directionI)angular.write(reinterpret_cast<const char*>(intensities[directionI].begin()),intensities[directionI].size()*sizeof(scalar));
'''
if 'scratch.gpu->solve' in text:
    export=export[:export.index('    forAll(intensities')]+"    std::vector<double> diagnosticIntensities;scratch.gpu->downloadIntensities(diagnosticIntensities);\n    angular.write(reinterpret_cast<const char*>(diagnosticIntensities.data()),diagnosticIntensities.size()*sizeof(double));\n"
text='#include <fstream>\n'+text.replace(needle,export+needle)
(out/'MieDoRadiationSolver.C').write_text(text)
PY
extra=()
if [[ $# == 4 ]]; then cuda=${CUDA_HOME:-/usr/local/cuda};extra=("$4" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart);fi
g++ -std=c++17 -O2 -ffp-contract=off -fopenmp -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \
 -I"$thermal" -I"$source/common" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" \
 -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" \
 -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" "$root/validation/gpu_coupled_pipeline/radiation_transport.cpp" \
 "$out/MieDoRadiationSolver.C" "$thermal/MieTable.C" "${extra[@]}" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream \
 -o "$out/radiation-test" > "$out/build.log" 2>&1
sha256sum "$out/radiation-test" "$thermal/MieDoRadiationSolver.C" "$thermal/MieTable.C" > "$out/binaries.sha256"
