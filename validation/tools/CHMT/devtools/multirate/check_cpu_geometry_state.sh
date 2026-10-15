#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
build=$(mktemp -d /tmp/chmt-cpu-geometry-state.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" -I"$validation" -I"$root/common")
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_cpu_wet_geometry.cpp" "$app/materials/MaterialTransport.C" "$app/mesh/Geometry.C" -o "$build/wet"
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_cpu_checkpoint.cpp" "$app/restart/Checkpoint.C" "$app/mesh/Geometry.C" -o "$build/checkpoint"
"$build/wet"
"$build/checkpoint"
