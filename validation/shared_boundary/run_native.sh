#!/usr/bin/env bash
# Source OpenFOAM 10 first. Run from any directory, with a new output directory.
set -euo pipefail
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
O=${1:?usage: run_native.sh NEW_OUTPUT_DIRECTORY}
mkdir "$O"
O=$(cd "$O" && pwd)
cd "$R"
python3 -m pytest -q -s validation/shared_boundary/test_math.py validation/shared_boundary/test_checkpoint.py validation/tests/gas_transport/test_unsupported_boundary_operator.py > "$O/unit.log" 2>&1
python3 validation/shared_boundary/native_smoke.py "$O/smoke"
python3 validation/shared_boundary/native_cases.py "$O/gas" gas
python3 validation/shared_boundary/native_cases.py "$O/coupled" coupled
python3 validation/shared_boundary/live_particles.py "$O/particles"
python3 validation/shared_boundary/extended.py "$O/extended"
if test -x build-evidence/shared-boundary/baseline-bin/gasUGKP; then
 python3 validation/shared_boundary/check_parity.py "$O/parity" "$O"
fi
mkdir "$O/build"
/usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_89 -Icommon validation/shared_boundary/resident.cu -o "$O/build/resident" > "$O/resident-build.log" 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode 9 "$O/build/resident" > "$O/resident-memcheck.log" 2>&1
FOAM_USER_APPBIN="$O/build" wmake validation/shared_boundary/native_reference > "$O/reference-build.log" 2>&1
"$O/build/boundaryReference" -case "$O/smoke/total" > "$O/reference.log" 2>&1
sha256sum bin/gasUGKP bin/gasUGKPSteady bin/FSH bin/FSHCudaBackend bin/CHT bin/CHMT "$O/build/resident" "$O/build/boundaryReference" > "$O/binaries.sha256"

python3 validation/shared_boundary/check_outputs.py "$O"
