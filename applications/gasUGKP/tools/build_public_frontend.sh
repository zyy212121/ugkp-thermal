#!/usr/bin/env bash
solver_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -euo pipefail
gas_species="${UGKWP_GAS_SPECIES-2}"
if [[ ! "${gas_species}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: UGKWP_GAS_SPECIES must be a positive integer" >&2
    exit 2
fi

cd "${solver_root}"
rm -f \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.o" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.o" \
    "Make/${WM_OPTIONS}/diluteUgkwpFoam.C.dep" \
    "Make/${WM_OPTIONS}/gpu/GpuBackendClient.C.dep"

UGKWP_CUDA_EXE_INC="-DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=${gas_species}" wmake

frontend="${FOAM_USER_APPBIN}/gasUGKP"
test -x "${frontend}"


if ldd "${frontend}" | grep -Eq 'libcuda|libcudart'; then
    echo "ERROR: public frontend directly links CUDA" >&2
    exit 1
fi
echo "Built ${frontend}"
