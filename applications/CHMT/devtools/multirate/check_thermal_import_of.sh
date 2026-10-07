#!/usr/bin/env bash
set -euo pipefail
[[ ${WM_PROJECT_VERSION:-} == 10 && -f ${WM_PROJECT_DIR:-/missing}/src/finiteVolume/lnInclude/fvCFD.H ]] || { echo 'BLOCKED: actual Foundation OpenFOAM 10 is required' >&2; exit 2; }
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-native-thermal-import.XXXXXX)
trap 'rm -rf "$build"' EXIT
"${CXX:-g++}" -std=c++14 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$app" -I"$app/../../common" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$app/devtools/multirate/test_thermal_import_of.C" "$app/materials/CpuMaterialDriver.C" "$app/materials/MaterialTransport.C" "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" "$app/mesh/Geometry.C" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$build/test"
mkdir -p "$build/case/constant"
cp -R "$app/devtools/multirate/of_case/system" "$build/case/system"
blockMesh -case "$build/case" > "$build/blockMesh.log"
"$build/test" -case "$build/case"
