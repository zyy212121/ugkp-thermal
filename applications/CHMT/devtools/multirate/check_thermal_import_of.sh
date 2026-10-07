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

# FULLDEBUG catches any accidental indexing of OpenFOAM's zero-sized Empty
# patch fields. This is an import-only fixture, not a 2-D solver validation.
"${CXX:-g++}" -std=c++14 -DFULLDEBUG -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$app" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$app/devtools/multirate/test_empty_import_of.C" "$app/mesh/Geometry.C" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$build/test_empty"
mkdir -p "$build/empty/constant"
cp -R "$app/devtools/multirate/of_case/system" "$build/empty/system"
cat > "$build/empty/system/blockMeshDict" <<'FOAM'
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (2 2 1) simpleGrading (1 1 1));
edges ();
boundary (walls { type wall; faces ((0 1 5 4) (3 7 6 2) (0 4 7 3) (1 2 6 5)); }
 frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); });
mergePatchPairs ();
FOAM
blockMesh -case "$build/empty" > "$build/blockMesh-empty.log"
"$build/test_empty" -case "$build/empty"
