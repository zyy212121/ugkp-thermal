#!/usr/bin/env bash
# Supplemental full OF10 coordinator exercise with sequential host execution of
# common kernel bodies. This is NOT native CUDA, device, or parallel-race evidence.
set -euo pipefail
[[ ${WM_PROJECT_VERSION:-} == 10 ]] || { echo 'Foundation OpenFOAM 10 environment required.' >&2; exit 2; }
app=$(cd "$(dirname "$0")/.." && pwd)
mode=();count=2;interval=.0005
case ${1:-} in
 '') ;;
 --chemistry) mode=(--chemistry);count=10;interval=.00001 ;;
 *) echo 'Usage: check_host_coupled_pipeline.sh [--chemistry]' >&2;exit 2 ;;
esac
if [[ -n ${CHMT_HOST_PIPELINE_EVIDENCE:-} ]]; then
 [[ ! -e $CHMT_HOST_PIPELINE_EVIDENCE ]] || { echo 'Host evidence destination already exists.' >&2; exit 2; }
 mkdir -p "$CHMT_HOST_PIPELINE_EVIDENCE";work=$(cd "$CHMT_HOST_PIPELINE_EVIDENCE" && pwd)
else
 work=$(mktemp -d /tmp/chmt-host-operator-pipeline.XXXXXX)
 trap 'rm -rf "$work"' EXIT
fi
python3 - "$app" "$work" <<'PY'
from pathlib import Path
import runpy, sys
app,work=map(Path,sys.argv[1:])
runpy.run_path(str(app/'tests/test_coupled_runtime_transactions.py'))['prepare_sequential_launches'](work)
(work/'CHMTBuildIdentity.H').write_text('#define CHMT_SOURCE_FINGERPRINT "host-emulation-not-native"\n#define CHMT_UPSTREAM_BASE "host-emulation-not-native"\n#define CHMT_BUILD_MANIFEST_JSON "{\\"artifact_kind\\":\\"HOST_OPERATOR_EMULATION_NOT_CUDA\\"}"\n')
PY
"${CXX:-g++}" -std=c++17 -O1 -DUGKWP_GAS_SPECIES="$count" -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \
 -I"$app" -I"$app/../../common" -I"$app/../../common/gasNumerics" -I"$work" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" \
 -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 "$app/CHMT.C" "$app/io/MultirateEvolution.C" "$app/materials/CpuMaterialDriver.C" \
 "$app/materials/MaterialTransport.C" "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" \
 "$app/mesh/Geometry.C" "$app/restart/Checkpoint.C" "$work/tree/applications/CHMT/gpu/Backend.cpp" \
 -L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream \
 -o "$work/CHMT-host-operator-emulation"
run_case(){
 local name=$1;shift
 python3 "$app/verification/coupled_runtime/make_case.py" "$work/$name" "$@"
 blockMesh -case "$work/$name" > "$work/$name/gas-mesh.log"
 blockMesh -case "$work/$name" -region solid -dict "$work/$name/system/blockMeshSolidDict" > "$work/$name/solid-mesh.log"
 "$work/CHMT-host-operator-emulation" -case "$work/$name" > "$work/$name/host-runtime.log" 2>&1
 cat "$work/$name/host-runtime.log"
 python3 "$app/verification/coupled_runtime/check_case.py" "$work/$name"
}
run_case full "${mode[@]}"
if [[ $count == 2 ]]; then
 run_case single --single
 run_case diffusive --diffusion
 run_case zero-diffusion --diffusion-control
 python3 "$app/verification/coupled_runtime/check_case.py" "$work/diffusive" --diffusion-control "$work/zero-diffusion"
else
 run_case disabled-chemistry --chemistry-control
 python3 "$app/verification/coupled_runtime/check_case.py" "$work/full" --chemistry-control "$work/disabled-chemistry"
fi
run_case split "${mode[@]}" --end-time "$interval"
run_case resumed "${mode[@]}" --restart "$work/split/chmtOutput/checkpoint-1"
python3 "$app/verification/coupled_runtime/check_case.py" "$work/resumed" --reference "$work/full" --restart-from "$work/split/chmtOutput/checkpoint-1"
printf '%s\n' 'PASS: supplemental OF10 coupled pipeline and restart with sequential HOST operators. Native CUDA remains NOT_RUN.'
