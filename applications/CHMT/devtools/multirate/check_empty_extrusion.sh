#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
bin=$(mktemp /tmp/chmt-empty-extrusion.XXXXXX)
trap 'rm -f "$bin"' EXIT
"${CXX:-g++}" -std=c++14 -O2 -Wall -Wextra -Werror -pedantic -I"$app" -I"$app/../../common" \
  "$app/devtools/multirate/test_empty_extrusion.cpp" "$app/mesh/Geometry.C" "$app/restart/Checkpoint.C" -o "$bin"
"$bin"
