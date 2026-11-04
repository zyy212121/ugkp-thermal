#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
bin=$(mktemp /tmp/chmt-empty-extrusion.XXXXXX)
trap 'rm -f "$bin"' EXIT
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I"$app" -I"$validation" -I"$app/../../common" \
  "$validation/devtools/multirate/test_empty_extrusion.cpp" "$app/mesh/Geometry.C" "$app/restart/Checkpoint.C" -o "$bin"
"$bin"
