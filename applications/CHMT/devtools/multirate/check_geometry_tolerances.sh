#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)
bin=$(mktemp /tmp/chmt-geometry-tolerances.XXXXXX)
trap 'rm -f "$bin"' EXIT
"${CXX:-g++}" -std=c++14 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$root/common" \
  "$root/applications/CHMT/devtools/multirate/test_geometry_tolerances.cpp" \
  "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin"
