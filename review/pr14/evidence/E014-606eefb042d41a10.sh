#!/bin/bash
set -eo pipefail
source /workspace/scratch/98b6339ec81b/chmt_implementation_20261011/native-environment/environment.sh
root=/workspace/scratch/98b6339ec81b/chmt_efficiency_work_20261011
app="$root/applications/CHMT"
build=/workspace/scratch/98b6339ec81b/chmt_efficiency_refresh_20261011/implementation/f2_owner
export TMPDIR="$build/tmp"
g++ --param ggc-min-expand=5 --param ggc-min-heapsize=32768 -DF2_TERMINAL=1 -std=c++17 -O0 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$app" -I"$root/common" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$app/tests/test_owner_endpoint_of.C" "$app/restart/Checkpoint.C" "$app/materials/MaterialTransport.C" "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" "$app/mesh/Geometry.C" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$build/green/endpoint_of"
mkdir -p "$build/case/constant";cp -R "$app/devtools/multirate/of_case/system" "$build/case/"
blockMesh -case "$build/case" >"$build/green/blockMesh.log"
"$build/green/endpoint_of" -case "$build/case"
