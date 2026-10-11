#!/usr/bin/env bash
# Native-only acceptance gate: no fake CUDA headers, host operator emulation,
# input-only binary, or pre-existing executable can substitute for this build.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
usage() { echo "Usage: $0 NEW_EVIDENCE_DIRECTORY [--single|--diffusion|--chemistry]" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
scenario=thermal
mode=()
case ${2:-} in
  '') ;;
  --single) scenario=single; mode=(--single) ;;
  --diffusion) scenario=diffusion; mode=(--diffusion) ;;
  --chemistry) scenario=chemistry; mode=(--chemistry) ;;
  *) usage ;;
esac
nvcc=${NVCC:-nvcc}
command -v "$nvcc" >/dev/null || { echo 'BLOCKED: native coupled acceptance requires CUDA toolkit nvcc; no host substitute is permitted.' >&2; exit 2; }
[[ ${WM_PROJECT_VERSION:-} == 10 ]] || { echo 'BLOCKED: source Foundation OpenFOAM 10 before native coupled acceptance.' >&2; exit 2; }
command -v blockMesh >/dev/null || { echo 'BLOCKED: actual OpenFOAM blockMesh is required.' >&2; exit 2; }
[[ ${CHMT_CUDA_ARCH:-} =~ ^sm_[0-9]+$ ]] || { echo 'BLOCKED: set CHMT_CUDA_ARCH to the supported native GPU architecture (sm_XX).' >&2; exit 2; }
[[ ! -e $1 ]] || { echo 'BLOCKED: evidence directory already exists; use a fresh path.' >&2; exit 2; }
mkdir -p "$1"
work=$(cd "$1" && pwd)
fixture="$app/verification/coupled_runtime"
export UGKWP_GAS_SPECIES=2
[[ $scenario != chemistry ]] || export UGKWP_GAS_SPECIES=10
export CHMT_BUILD="$app/.build/native-acceptance-$scenario"
"$nvcc" -std=c++17 -arch="$CHMT_CUDA_ARCH" "$fixture/cuda_probe.cu" -o "$work/native-cuda-probe"
"$work/native-cuda-probe" | tee "$work/native-cuda-probe.log"
bash "$app/Allwmake" > "$work/build.log" 2>&1
cp "$CHMT_BUILD/generated/build.json" "$work/build.json"
python3 - "$work/build.json" "$UGKWP_GAS_SPECIES" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1]))
assert manifest['artifact_kind']=='CHMT_BUILD_MANIFEST', 'not a native build manifest'
assert manifest['Ns']==int(sys.argv[2]) and manifest['precision']=='FP64'
assert manifest['cuda_arch'].startswith('sm_') and 'COMPILE_ONLY' not in manifest['cuda_compiler']
PY
run_case() {
  local name=$1; shift
  python3 "$fixture/make_case.py" "$work/$name" "$@"
  blockMesh -case "$work/$name" > "$work/$name/gas-mesh.log" 2>&1
  blockMesh -case "$work/$name" -region solid -dict "$work/$name/system/blockMeshSolidDict" > "$work/$name/solid-mesh.log" 2>&1
  "$CHMT_BUILD/CHMT" -case "$work/$name" > "$work/$name/native-runtime.log" 2>&1
  grep -q 'CHMT accepted time=' "$work/$name/native-runtime.log"
  python3 "$fixture/check_case.py" "$work/$name" > "$work/$name/independent-check.json"
}
run_case full "${mode[@]}"
if [[ $scenario == single ]]; then
  # Runtime constant switch, same exact Ns2 executable, no intervening build.
  run_case frozen-companion
fi
interval=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coupling_interval"])' "$work/full/verification.json")
run_case split "${mode[@]}" --end-time "$interval"
checkpoint=$(python3 - "$work/split/chmtOutput" <<'PY'
from pathlib import Path
import sys
print(max(Path(sys.argv[1]).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])))
PY
)
run_case resumed "${mode[@]}" --restart "$checkpoint"
python3 "$fixture/check_case.py" "$work/resumed" --reference "$work/full" --restart-from "$checkpoint" > "$work/restart-check.json"
if [[ $scenario != chemistry ]]; then
  run_case refined "${mode[@]}" --refine
  python3 "$fixture/check_case.py" "$work/full" --refinement "$work/refined" > "$work/refinement-check.json"
else
  run_case disabled-chemistry --chemistry-control
  python3 "$fixture/check_case.py" "$work/full" --chemistry-control "$work/disabled-chemistry" > "$work/chemistry-check.json"
fi
if [[ $scenario == diffusion ]]; then
  run_case zero-diffusion --diffusion-control
  python3 "$fixture/check_case.py" "$work/full" --diffusion-control "$work/zero-diffusion" > "$work/diffusion-check.json"
fi
python3 - "$work" "$scenario" "$CHMT_BUILD/CHMT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); binary=Path(sys.argv[3])
report={'evidence':'native-cuda-coupled-runtime','scenario':sys.argv[2],
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'build':json.loads((root/'build.json').read_text()),
        'cuda_probe':(root/'native-cuda-probe.log').read_text(),
        'independent_checks':json.loads((root/'full/independent-check.json').read_text()),
        'restart':json.loads((root/'restart-check.json').read_text())}
if (root/'frozen-companion/independent-check.json').exists():
    report['same_binary_frozen_companion']=json.loads((root/'frozen-companion/independent-check.json').read_text())
for check in ('diffusion', 'chemistry', 'refinement'):
    path=root/(check+'-check.json')
    if path.exists(): report[check]=json.loads(path.read_text())
(root/'native-acceptance.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('PASS: native CUDA coupled '+sys.argv[2]+' evolution, independent conservation/exchange, and restart continuation. Evidence: '+str(root/'native-acceptance.json'))
PY
