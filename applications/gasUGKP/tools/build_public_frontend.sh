#!/usr/bin/env bash
solver_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail

cd "${solver_root}"
rm -f \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.o" \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.C.dep"

UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA" wmake

frontend="${FOAM_USER_APPBIN}/gasUGKP"
test -x "${frontend}"


if ldd "${frontend}" | grep -Eq 'libcuda|libcudart'; then
    echo "ERROR: public frontend directly links CUDA" >&2
    exit 1
fi
echo "Built ${frontend}"
