#!/usr/bin/env bash
# Native regression for the reviewed total-pressure and RK boundary-time fixes.
set -euo pipefail
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
O=${1:?usage: run_review_regression.sh NEW_OUTPUT_DIRECTORY}
mkdir "$O"
O=$(cd "$O" && pwd)
if [[ -z ${WM_PROJECT_DIR:-} ]]; then
    set +u
    source "${OPENFOAM_BASHRC:-/opt/openfoam10/etc/bashrc}"
    set -u
fi
cd "$R"
git rev-parse HEAD > "$O/source-commit.txt"
git diff --binary > "$O/source-working-tree.patch"
NVCC_PREPEND_FLAGS=-DUGKP_BOUNDARY_FLUX_TRACE bash tool/build_local.sh > "$O/trace-build.log" 2>&1
sha256sum bin/gasUGKP bin/gasUGKPSteady bin/FSH bin/FSHCudaBackend bin/CHT bin/CHMT > "$O/trace-binaries.sha256"
python3 validation/shared_boundary/stage_flux_regression.py "$O/stages" > "$O/stages.log" 2>&1
python3 validation/shared_boundary/compare_total_pressure.py "$O/total-pressure" > "$O/total-pressure.log" 2>&1
python3 -m pytest -q validation/shared_boundary/test_math.py     validation/tests/gas_transport/test_mixture_wall_function.py     validation/tests/gas_transport/test_review_sst_wall_audits.py     validation/composition_transport/test_configuration.py > "$O/unit.log" 2>&1
python3 validation/composition_transport/pr_wall_regression.py "$O/pr-wall-host" > "$O/pr-wall-host.log" 2>&1
python3 validation/composition_transport/pr_wall_regression.py "$O/pr-wall-cuda" --cuda > "$O/pr-wall-cuda.log" 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99     "$O/pr-wall-cuda/probe" > "$O/pr-wall-memcheck.log" 2>&1
# Leave the checkout with ordinary production executables, without tracing.
env -u NVCC_PREPEND_FLAGS bash tool/build_local.sh > "$O/production-build.log" 2>&1
python3 validation/shared_boundary/native_smoke.py "$O/smoke" > "$O/smoke.log" 2>&1
python3 validation/shared_boundary/native_cases.py "$O/chmt" CHMT > "$O/chmt.log" 2>&1
sha256sum bin/gasUGKP bin/gasUGKPSteady bin/FSH bin/FSHCudaBackend bin/CHT bin/CHMT     "$O/total-pressure/build/total-pressure" "$O/total-pressure/build/boundaryReference"     "$O/pr-wall-host/probe" "$O/pr-wall-cuda/probe" > "$O/production-binaries.sha256"
printf 'PASS\n' > "$O/completed.txt"
