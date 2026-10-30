#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
bin=$(mktemp /tmp/chmt-sweep-constraints.XXXXXX)
trap 'rm -f "$bin"' EXIT
for test in test_sweep_constraints test_sweep_remainder test_dense_curved_remainder; do
"${CXX:-g++}" -std=c++17 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$validation" -I"$root/common" \
  "$validation/devtools/multirate/$test.cpp" \
  "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin"
done
