#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --stress ) ]]; then
  echo "usage: bash check_large_sweep_constraints.sh [--stress]" >&2
  exit 2
fi
tests="$validation/devtools/multirate"
bin=$(mktemp /tmp/chmt-large-sweep-constraints.XXXXXX)
trap 'rm -f "$bin"' EXIT
"${CXX:-g++}" -std=c++14 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$validation" -I"$root/common" \
  "$tests/test_large_sweep_constraints.cpp" \
  "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
status=0
python3 "$tests/test_harmonic_output_traversal.py" || status=1
# Separate processes make every former default-cap regression visible in RED.
for fixture in hex130 tet130 tet880 near-rank nonlinear controls trajectory harmonic allocator; do
  "$bin" "$fixture" || status=1
done
if [[ ${1:-} == --stress ]]; then
  "$bin" tet4500 || status=1
fi
# Keep deliberate global new/delete fault interposition uninlined. GCC
# diagnoses the malloc/free replacement pair as mismatched after O2 inlining.
"${CXX:-g++}" -std=c++14 -O0 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$validation" -I"$root/common" \
  "$tests/test_sweep_workspace_faults.cpp" "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin" || status=1
# Reuse the original target/trajectory, Empty pairing, moving and restart tests.
for suite in check_sweep_constraints check_empty_extrusion check_cpu_geometry_state; do
  bash "$runners/devtools/multirate/$suite.sh" || status=1
done
exit "$status"
