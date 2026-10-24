#!/usr/bin/env bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${OF10_ENV:-}" ]; then source "$OF10_ENV" || exit $?; fi
: "${WM_PROJECT_DIR:?Source OpenFOAM10 or set OF10_ENV}"
export DRY_CORE_ROOT="${DRY_CORE_ROOT:-$(cd "$HERE/../../.." && pwd)}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$HERE/results" "$HERE/bin" "$HERE/case/system" "$HERE/case/constant"
app="$DRY_CORE_ROOT/applications/CHMT"
flags=(-std=c++17 -O0 -g0 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -DUGKWP_GAS_SPECIES=2
 -I"$app" -I"$DRY_CORE_ROOT/common")
for inc in OpenFOAM finiteVolume meshTools OSspecific/POSIX; do flags+=(-I"$WM_PROJECT_DIR/src/$inc/lnInclude"); done
objects=()
: > "$HERE/results/build.log"
for source in "$HERE/nativeDryOwner.C" "$app/materials/CpuMaterialDriver.C" "$app/materials/MaterialTransport.C" "$app/ablation/CpuSurfaceInterface.C" "$app/mesh/Geometry.C"; do
 object="$HERE/bin/$(basename "$source" .C).o"
 if [ ! -f "$object" ] || [ "$source" -nt "$object" ] || [ "$HERE/run_adapter.sh" -nt "$object" ] || [ -n "$(find "$app" "$DRY_CORE_ROOT/common" -type f \( -name '*.H' -o -name '*.h' -o -name '*.cuh' \) -newer "$object" -print -quit 2>/dev/null)" ]; then
  "${CXX:-g++}" "${flags[@]}" -c "$source" -o "$object" >> "$HERE/results/build.log" 2>&1
 fi
 objects+=("$object")
done
"${CXX:-g++}" "${objects[@]}" -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM \
 -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$HERE/bin/nativeDryOwner" >> "$HERE/results/build.log" 2>&1
python3 - "$HERE/case" <<'PY'
from pathlib import Path
import sys
case=Path(sys.argv[1]);header='FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n'
files={
'controlDict':'application nativeDryOwner; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .01; writeControl timeStep; writeInterval 1; runTimeModifiable false;',
'fvSchemes':'ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear orthogonal; } interpolationSchemes { default linear; } snGradSchemes { default orthogonal; }',
'fvSolution':'solvers {}',
'blockMeshDict':'convertToMeters 1; vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1)); blocks (hex (0 1 2 3 4 5 6 7) (2 2 2) simpleGrading (1 1 1)); edges (); boundary (left { type wall; faces ((0 4 7 3)); } right { type wall; faces ((1 2 6 5)); } sides { type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }); mergePatchPairs ();'}
for name,body in files.items():(case/'system'/name).write_text(header%name+body+'\n')
PY
blockMesh -case "$HERE/case" > "$HERE/results/blockMesh.log" 2>&1
"$HERE/bin/nativeDryOwner" -case "$HERE/case" > "$HERE/results/run.log" 2>&1
cat "$HERE/results/run.log"
