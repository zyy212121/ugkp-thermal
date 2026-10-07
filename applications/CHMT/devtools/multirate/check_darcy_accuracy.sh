#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-darcy-accuracy.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-O2 -std=c++14 -Wall -Wextra -Werror -pedantic -I"$app")
if [[ ${CHMT_SANITIZE:-0} == 1 ]]; then
    flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_material_darcy_3d.cpp" \
    "$app/materials/MaterialTransport.C" -o "$build/darcy"
"$build/darcy"
# This checks retained CUDA source wiring, not CUDA compilation or execution.
python3 - "$app" <<'PY'
from pathlib import Path
import sys
app = Path(sys.argv[1])
material = (app/'materials/MaterialKernels.cu').read_text()
assert 'darcyNormalPressureGradient(' in material
assert 'darcyConductionFluxWithGradient(' in material
assert 'materialReconstructionLimiter(value,lo,hi,change)' in material
assert '*((b.pressure-a.pressure)/distance-' not in material
assert 'solid.gradientPressure[cell]=gradientP;' in material
assert 'solid.pressureGradientSensitivity[cell]=sensitivity;' in material
assert '!solid.gradientPressure' in material and '!solid.pressureGradientSensitivity' in material
resources = (app/'gpu/BackendResources.H').read_text()
for name in ('solidPressureGradient', 'solidPressureGradientSensitivity'):
    assert name+'.rebindFault(fault);' in resources
    assert 'CHMT_ALLOC('+name+',ns)' in resources
    assert name+'.data()' in resources
backend = (app/'gpu/Backend.cu').read_text()
assert 'darcyPressureStencilInverseLength(' in backend
print('PASS: retained CUDA Darcy gradient/allocation/stiffness source wiring (not native CUDA compilation)', file=sys.stderr)
PY
