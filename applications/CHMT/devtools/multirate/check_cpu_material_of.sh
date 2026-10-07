#!/usr/bin/env bash
set -euo pipefail
[[ ${WM_PROJECT_VERSION:-} == 10 && -f ${WM_PROJECT_DIR:-/missing}/src/finiteVolume/lnInclude/fvCFD.H ]] || { echo 'BLOCKED: real Foundation OpenFOAM 10 headers/runtime required; no mock solver fallback.' >&2; exit 2; }
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-native-material.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
"$cxx" -std=c++14 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$app" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$app/devtools/multirate/test_cpu_material_of.C" "$app/materials/CpuMaterialDriver.C" "$app/materials/MaterialTransport.C" "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" "$app/mesh/Geometry.C" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$build/test_cpu_material_of"
[[ $# -le 1 ]] || { echo 'Usage: check_cpu_material_of.sh [actual-2x2x2-case]' >&2; exit 2; }
caseDir=${1:-$build/case}
if [[ $# == 0 ]]; then
  mkdir -p "$caseDir/constant"
  cp -R "$app/devtools/multirate/of_case/system" "$caseDir/system"
  blockMesh -case "$caseDir" > "$build/blockMesh.log"
fi
"$build/test_cpu_material_of" -case "$caseDir"
