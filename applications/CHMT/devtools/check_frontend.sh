#!/usr/bin/env bash
# Actual OpenFOAM frontend execution, limited to early coupling input rejection.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
bash "$app/Allwmake" --frontend-only
binary=${CHMT_FRONTEND_BUILD:-$app/.build/frontend}/CHMT-input-check
work=$(mktemp -d /tmp/chmt-coupling-entry.XXXXXX)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/case/constant"
cp -R "$app/devtools/multirate/of_case/system" "$work/case/system"
for mode in Standalone LegacyExplicit; do
  printf 'FoamFile { version 2.0; format ascii; class dictionary; object chmtProperties; }\nexecutionMode %s;\n' "$mode" > "$work/case/constant/chmtProperties"
  if "$binary" -case "$work/case" -check-input > "$work/result" 2>&1; then
    echo "FAIL: native CHMT frontend accepted $mode" >&2; exit 1
  fi
  grep -q 'requires executionMode Multirate' "$work/result"
done
printf 'FoamFile { version 2.0; format ascii; class dictionary; object chmtProperties; }\nexecutionMode Multirate;\n' > "$work/case/constant/chmtProperties"
if "$binary" -case "$work/case" -check-input > "$work/result" 2>&1; then
  echo 'FAIL: native CHMT frontend accepted missing coupled participants' >&2; exit 1
fi
grep -q 'requires a material region and mapped surface' "$work/result"
# Positive case: real gas and solid meshes with nonzero thermal contrast and
# a conformal interface. This validates input ownership, not time evolution.
python3 "$app/verification/coupled_input/make_case.py" "$work/coupled"
blockMesh -case "$work/coupled" > "$work/gasMesh.log"
blockMesh -case "$work/coupled" -region solid -dict "$work/coupled/system/blockMeshSolidDict" > "$work/solidMesh.log"
"$binary" -case "$work/coupled" -check-input > "$work/valid.log" 2>&1
grep -q 'CHMT coupled input valid' "$work/valid.log"
if "$binary" -case "$work/coupled" > "$work/runtime.log" 2>&1; then
  echo 'FAIL: unfinished shared backend claimed native coupled evolution' >&2; exit 1
fi
grep -q 'input-only build has no CUDA backend' "$work/runtime.log"
# Same compiled Ns binary, explicitly selected single-component transport.
python3 "$app/verification/coupled_input/make_case.py" "$work/single" --single
blockMesh -case "$work/single" > "$work/singleGasMesh.log"
blockMesh -case "$work/single" -region solid -dict "$work/single/system/blockMeshSolidDict" > "$work/singleSolidMesh.log"
"$binary" -case "$work/single" -check-input > "$work/singleValid.log" 2>&1
grep -q 'CHMT coupled input valid' "$work/singleValid.log"
"$binary" -build-info > "$work/build-info.log" 2>&1
grep -q 'COMPILE_ONLY' "$work/build-info.log"
printf '%s\n' 'PASS: actual OF10 frontend rejects uncoupled modes and accepts a real two-region input. Input-only binary refuses evolution explicitly.'
