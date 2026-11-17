#!/usr/bin/env bash
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
if [[ -z ${WM_PROJECT_DIR:-} ]]; then
    source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
fi
set -euo pipefail
build="${UGKP_LAUNCHER_BUILD:-$source_root/.build/precision/launcher}"
mkdir -p "$build"
"${CXX:-g++}" -std=c++17 -O2 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 \
    -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" \
    -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \
    "$source_root/tool/precision/PrecisionLauncher.C" \
    -L"$FOAM_LIBBIN" -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream \
    -o "$build/ugkpPrecisionLauncher"
printf '%s\n' "$build/ugkpPrecisionLauncher"
