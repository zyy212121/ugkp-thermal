#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
bin=$(mktemp /tmp/chmt-geometry-tolerances.XXXXXX)
trap 'rm -f "$bin"' EXIT
"${CXX:-g++}" -std=c++17 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$validation" -I"$root/common" \
  "$validation/devtools/multirate/test_geometry_tolerances.cpp" \
  "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin"
