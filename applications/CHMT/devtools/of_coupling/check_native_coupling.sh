#!/usr/bin/env bash
set -euo pipefail
[[ ${WM_PROJECT_VERSION:-} == 10 && -f ${WM_PROJECT_DIR:-/missing}/applications/solvers/compressible/rhoCentralFoam/directionInterpolate.H ]] || { echo 'BLOCKED: real Foundation OpenFOAM 10 runtime required' >&2; exit 2; }
app=$(cd "$(dirname "$0")/../.." && pwd)
evidence=${1:-$(mktemp -d /tmp/chmt-native-coupling.XXXXXX)}
mkdir -p "$evidence"
export UCX_VFS_ENABLE=n
export PYTHONDONTWRITEBYTECODE=1
{ echo "WM_PROJECT_VERSION=$WM_PROJECT_VERSION"; echo "WM_PROJECT_DIR=$WM_PROJECT_DIR"; "${CXX:-g++}" --version | head -1; } > "$evidence/environment.txt"
find "$app" -type f \( -name "*.H" -o -name "*.C" -o -name "*.py" -o -name "*.sh" \) -print0 | sort -z | xargs -0 sha256sum > "$evidence/source-sha256.txt"
cxx=${CXX:-g++}
"$cxx" -O1 -std=c++14 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$app" -I"$app/../../common" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 -I"$WM_PROJECT_DIR/applications/solvers/compressible/rhoCentralFoam" \
 "$app/devtools/of_coupling/native_coupling.C" "$app/materials/CpuMaterialDriver.C" "$app/materials/MaterialTransport.C" "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" "$app/mesh/Geometry.C" "$app/restart/Checkpoint.C" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$evidence/chmtNativeOfCoupling" > "$evidence/build.log" 2>&1
for spec in 'incompatible 0.0000004 0.000004' 'gcl 0.0000002 0.000002' 'moving 0.0000001 0.000002' 'moving 0.0000001 0.000001' 'fixed 0.0000002 0.000002' 'moving 0.0000004 0.000004' 'moving 0.0000002 0.000004' 'moving 0.0000001 0.000004'; do
 read -r mode dt window <<< "$spec"
 caseDir="$evidence/${mode}_${dt}_${window}"
 blocks=4; [[ "$mode" == incompatible ]] && blocks=12
 python3 "$app/devtools/of_coupling/make_cases.py" "$caseDir" "$blocks"
 blockMesh -case "$caseDir" > "$caseDir/blockMesh.log" 2>&1
 blockMesh -case "$caseDir" -region solid -dict system/blockMeshSolidDict > "$caseDir/blockMeshSolid.log" 2>&1
 if [[ "$mode" != incompatible ]]; then
  python3 "$app/devtools/of_coupling/make_tetra_mesh.py" "$caseDir" 4 > "$caseDir/tetraMesh.log"
  checkMesh -case "$caseDir" > "$caseDir/checkMesh.log" 2>&1
  checkMesh -case "$caseDir" -region solid > "$caseDir/checkMeshSolid.log" 2>&1
  grep -q "Mesh OK" "$caseDir/checkMesh.log"
  grep -q "Mesh OK" "$caseDir/checkMeshSolid.log"
 fi
 "$evidence/chmtNativeOfCoupling" -case "$caseDir" -mode "$mode" -micro "$dt" -window "$window" > "$caseDir/run.log" 2>&1
 if [[ "$mode" != incompatible ]]; then
  checkMesh -case "$caseDir" -latestTime > "$caseDir/checkMeshFinal.log" 2>&1
  grep -q "Mesh OK" "$caseDir/checkMeshFinal.log"
  if [[ "$mode" != gcl ]]; then
   checkMesh -case "$caseDir" -region solid -latestTime > "$caseDir/checkMeshSolidFinal.log" 2>&1
   grep -q "Mesh OK" "$caseDir/checkMeshSolidFinal.log"
  fi
 fi
 done
python3 "$app/devtools/of_coupling/check_results.py" "$evidence"
sha256sum --check "$evidence/source-sha256.txt" > "$evidence/source-verification-after.log"
echo "Native OF10 coupling evidence: $evidence"
