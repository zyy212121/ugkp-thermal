#!/usr/bin/env bash
solver_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail
gas_species="${UGKWP_GAS_SPECIES-2}"

cd "${solver_root}"
rm -f \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.o" \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.C.dep"

UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=${gas_species}" wmake

frontend="${FOAM_USER_APPBIN}/gasUGKP"
test -x "${frontend}"


echo "Built ${frontend}"
