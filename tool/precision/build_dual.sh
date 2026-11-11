#!/usr/bin/env bash
# Each precision owns a complete source, object, archive and executable tree.
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
if [[ -z ${WM_PROJECT_DIR:-} ]]; then
    source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
fi
set -euo pipefail
output_bin="${FOAM_USER_APPBIN:-$source_root/bin}"
case "$output_bin" in /*) ;; *) output_bin="$PWD/$output_bin";; esac
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NVCC="$CUDA_HOME/bin/nvcc"
export WM_NCOMPPROCS="${WM_NCOMPPROCS:-1}"
solvers=(gasUGKP gasUGKPSteady CHT CHMT)
selected_solver=""
if [[ ${1:-} == --solver && $# == 2 ]]; then
    case "$2" in
        gasUGKP|gasUGKPSteady|CHT|CHMT) selected_solver="$2"; solvers=("$2");;
        *) echo "Unknown solver: $2" >&2; exit 2;;
    esac
elif [[ $# != 0 ]]; then
    echo "Usage: build_dual.sh [--solver gasUGKP|gasUGKPSteady|CHT|CHMT]" >&2
    exit 2
fi
python3 "$source_root/tool/precision/report_overlay_pairs.py" --source-root "$source_root" \
    --output "$source_root/.build/precision/overlay-pairs.json" \
    || echo "Warning: overlay pair report unavailable; build is not blocked." >&2
for bits in 64 32; do
    tree=$(python3 "$source_root/tool/precision/materialize_sources.py" \
        --source-root "$source_root" --precision "$bits")
    echo "=== Building independent FP$bits solver tree: $tree ==="
    (
        export UGKP_INTERNAL_PRECISION_BITS="$bits"
        export FOAM_USER_APPBIN="$tree/bin"
        export FOAM_USER_LIBBIN="$tree/lib"
        export GAS_UGKP_BUILD="$tree/applications/gasUGKP/private_backend/build/$WM_OPTIONS"
        export GAS_UGKP_STEADY_BUILD="$tree/applications/gasUGKPSteady/private_backend/build/$WM_OPTIONS"
        export CHMT_BUILD="$tree/applications/CHMT/.build/native"
        export CHMT_FRONTEND_BUILD="$tree/applications/CHMT/.build/frontend"
        export UGKWP_CUDA_EXE_INC="-I$CUDA_HOME/include -DUGKWP_USE_CUDA"
        mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
        if [[ -z "$selected_solver" ]]; then
            bash "$tree/Allwmake"
        elif [[ "$selected_solver" == CHMT ]]; then
            bash "$tree/applications/CHMT/Allwmake"
        else
            bash "$tree/tool/$selected_solver/build_cuda_solver.sh"
        fi
    )
done
bash "$source_root/tool/precision/build_launcher.sh"
launcher="${UGKP_LAUNCHER_BUILD:-$source_root/.build/precision/launcher}/ugkpPrecisionLauncher"
# Publish only after both complete builds and the launcher have succeeded.
mkdir -p "$output_bin/.precision/32" "$output_bin/.precision/64"
for bits in 64 32; do
    tree="$source_root/.build/precision/fp$bits/source"
    for solver in "${solvers[@]}"; do
        install -m 755 "$tree/bin/$solver" "$output_bin/.precision/$bits/$solver"
    done
done
for solver in "${solvers[@]}"; do
    install -m 755 "$launcher" "$output_bin/$solver"
done
echo "=== Dual-precision executables ==="
for solver in "${solvers[@]}"; do
    sha256sum "$output_bin/$solver" "$output_bin/.precision/32/$solver" "$output_bin/.precision/64/$solver"
done
