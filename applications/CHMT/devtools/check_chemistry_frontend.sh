#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
export UGKWP_GAS_SPECIES=10
export CHMT_FRONTEND_BUILD=${CHMT_FRONTEND_BUILD:-$app/.build/frontend-h2o2}
bash "$app/Allwmake" --frontend-only
work=$(mktemp -d /tmp/chmt-reacting-input.XXXXXX)
trap 'rm -rf "$work"' EXIT
python3 "$app/verification/coupled_input/make_case.py" "$work/case" --h2o2
blockMesh -case "$work/case" > "$work/gasMesh.log"
blockMesh -case "$work/case" -region solid -dict "$work/case/system/blockMeshSolidDict" > "$work/solidMesh.log"
"$CHMT_FRONTEND_BUILD/CHMT-input-check" -case "$work/case" -check-input > "$work/input.log" 2>&1
grep -q 'CHMT coupled input valid' "$work/input.log"
printf '%s\n' 'PASS: actual OF10 coupled preflight accepts canonical 10-species/29-reaction chemistry model. No GPU kinetics or coupled evolution was executed.'
