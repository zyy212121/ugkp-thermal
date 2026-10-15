#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
cases_root="${UGKP_CASES_ROOT:-$(dirname "$root")/ugkp-thermal-cases}"
export UGKWP_GAS_SPECIES=10
export CHMT_FRONTEND_BUILD=${CHMT_FRONTEND_BUILD:-$app/.build/frontend-h2o2}
bash "$app/Allwmake" --frontend-only
work=$(mktemp -d /tmp/chmt-reacting-input.XXXXXX)
trap 'rm -rf "$work"' EXIT
python3 "$cases_root/test/tools/CHMT/coupled_input/make_case.py" "$work/case" --h2o2
blockMesh -case "$work/case" > "$work/gasMesh.log"
blockMesh -case "$work/case" -region solid -dict "$work/case/system/blockMeshSolidDict" > "$work/solidMesh.log"
"$CHMT_FRONTEND_BUILD/CHMT-input-check" -case "$work/case" -check-input > "$work/input.log" 2>&1
grep -q 'CHMT coupled input valid' "$work/input.log"
printf '%s\n' 'PASS: actual OF10 coupled preflight accepts canonical 10-species/29-reaction chemistry model. No GPU kinetics or coupled evolution was executed.'
