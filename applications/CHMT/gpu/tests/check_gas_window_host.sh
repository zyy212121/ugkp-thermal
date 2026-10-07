#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
"$cxx" -std=c++14 -Wall -Wextra -Werror -pedantic -I"$root/applications/CHMT" \
    "$root/applications/CHMT/gpu/tests/GasWindowHostTests.C" -o "$build/gas_window_host"
"$build/gas_window_host"
# The following checks ownership and host-only arithmetic. It is not CUDA
# compilation, a GPU run, or an end-to-end coupled conservation test.
python3 - "$root" <<'PY'
import pathlib, sys
root=pathlib.Path(sys.argv[1])
s=(root/'applications/CHMT/gpu/GasWindowImplementation.cuh').read_text()
for forbidden in ('launchMaterialRhs(', 'launchFilmRhs(', 'launchInterfacePackets(', 'launchParticipantUpdate(', '.solid.download(', '.film.download('):
    assert forbidden not in s, forbidden
stable=(root/'applications/CHMT/gpu/GasStability.H').read_text()
for forbidden in ('solid.', 'film.', 'permeability', 'reactionRate', 'liquid.'):
    assert forbidden not in stable, forbidden
assert 'gasScale*in.bulk.pressure*in.normalSpeed' not in (root/'applications/CHMT/gpu/GasWallMath.H').read_text()
assert 'in.bulk.pressure*in.sweptVolume' in (root/'applications/CHMT/gpu/GasWallMath.H').read_text()
assert 'if(window.staticGeometry)' in s
assert 'maximumHistoryBytes' in s and 'maximumRecords' in s
print('gas-only ownership/source checks passed (CUDA execution not tested)')
PY
