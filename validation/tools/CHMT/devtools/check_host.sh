#!/usr/bin/env bash
# CPU contract/component evidence only. This is not the native GPU acceptance gate.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
build=$(mktemp -d /tmp/chmt-rebuild-host.XXXXXX)
trap 'rm -rf "$build"' EXIT
flags=(-std=c++14 -O2 -Wall -Wextra -Werror -pedantic -I"$app" -I"$validation" -I"$app/../../common")
for test in core transaction; do
  "${CXX:-g++}" "${flags[@]}" "$validation/tests/test_$test.cpp" -o "$build/$test"
  "$build/$test"
done
for test in interval_contracts interval_audit; do
  "${CXX:-g++}" "${flags[@]}" "$validation/verification/multirate/$test.cpp" -o "$build/$test"
  "$build/$test"
done
"${CXX:-g++}" "${flags[@]}" "$validation/tests/test_multirate_orchestration.cpp" "$app/io/MultirateEvolution.C" -o "$build/orchestration"
"$build/orchestration"
"${CXX:-g++}" "${flags[@]}" "$validation/tests/test_restart.cpp" "$app/restart/Checkpoint.C" "$app/mesh/Geometry.C" -Wl,--wrap=fsync -o "$build/restart"
"$build/restart"
for check in check_cpu_material check_cpu_film check_geometry_tolerances check_sweep_constraints check_large_sweep_constraints check_empty_extrusion check_cpu_geometry_state check_darcy_accuracy check_reaction_accuracy; do
  bash "$runners/devtools/multirate/$check.sh"
done
python3 -m pytest -q "$validation/tests/test_coupling_architecture.py" "$validation/tests/test_shared_gas_bindings.py" "$validation/tests/test_shared_thermo.py" "$validation/tests/test_shared_device_storage.py" "$validation/tests/test_shared_backend_compile.py" "$validation/tests/test_coupled_runtime_transactions.py" "$validation/tests/test_restoration_identity.py"
printf '%s\n' 'PASS: selected CHMT CPU contracts/components only. Native shared GPU coupled backend remains an unverified integration requirement.'
