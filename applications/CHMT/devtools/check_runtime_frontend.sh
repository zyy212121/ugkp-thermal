#!/usr/bin/env bash
# Parse the actual native branch without linking or claiming a CUDA build.
set -euo pipefail
[[ ${WM_PROJECT_VERSION:-} == 10 ]] || { echo 'Foundation OpenFOAM 10 environment required.' >&2; exit 2; }
app=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d /tmp/chmt-runtime-syntax.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/CHMTBuildIdentity.H" <<'EOF'
// Test-only identities for a syntax-only translation. Never linked or executed.
#define CHMT_BUILD_MANIFEST_JSON "syntax-only-test"
#define CHMT_SOURCE_FINGERPRINT "syntax-only-test"
#define CHMT_UPSTREAM_BASE "syntax-only-test"
EOF
"${CXX:-g++}" -std=c++17 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -DUGKWP_GAS_SPECIES="${UGKWP_GAS_SPECIES:-2}" \
 -I"$app" -I"$app/../../common" -I"$scratch" \
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" \
 -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
 -fsyntax-only "$app/CHMT.C" "$app/io/MultirateEvolution.C"
printf '%s\n' 'PASS: actual native CHMT frontend/coordinator C++ branches parse against OF10. CUDA build and execution were not performed.'
