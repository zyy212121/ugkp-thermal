#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
root=$(cd "$app/../.." && pwd)
build=$(mktemp -d /tmp/chmt-cpu-geometry-state.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" -I"$root/common")
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_cpu_wet_geometry.cpp" "$app/materials/MaterialTransport.C" "$app/mesh/Geometry.C" -o "$build/wet"
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_cpu_checkpoint.cpp" "$app/restart/Checkpoint.C" "$app/mesh/Geometry.C" -o "$build/checkpoint"
"$build/wet"
"$build/checkpoint"
