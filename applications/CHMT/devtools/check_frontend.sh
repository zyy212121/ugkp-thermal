#!/usr/bin/env bash
# Actual OF10 COMPILE_ONLY object check: never links a fake backend or runs CFD.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
app=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ ${WM_PROJECT_VERSION:-} == 10 && -f ${WM_PROJECT_DIR:-/missing}/src/finiteVolume/lnInclude/fvCFD.H ]] || { echo 'Source Foundation OpenFOAM 10 headers/runtime first' >&2; exit 2; }
build=$(mktemp -d /tmp/chmt-frontend-compile.XXXXXX)
trap 'rm -rf "$build"' EXIT
python3 "$app/devtools/build_info.py" --compile-only --output "$build" --species "${CHMT_SPECIES:-S0,S1}" --cxx "${CXX:-g++}"
"${CXX:-g++}" -std=c++14 -Wall -Wextra -Wno-unused-parameter -fPIC -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -DCHMT_FRONTEND_COMPILE_ONLY '-DCHMT_SPECIES_HEADER="CHMTSpecies.H"' \
  -I"$app" -I"$app/../../common" -I"$build" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" \
  -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" \
  -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" -c "$app/CHMT.C" -o "$build/CHMT.o"
printf '%s\n' 'COMPILE_ONLY: actual Foundation OpenFOAM 10 CHMT frontend object compiled. No executable, CUDA compile/link or CFD run was performed.'
