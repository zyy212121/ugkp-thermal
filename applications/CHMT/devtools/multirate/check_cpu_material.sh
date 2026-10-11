#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-cpu-material.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" -I"$app/../../common")
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_cpu_material.cpp" -o "$build/caloric"
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_material_transport.cpp" "$app/materials/MaterialTransport.C" -o "$build/transport"
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_cpu_surface.cpp" "$app/ablation/CpuSurfaceInterface.C" -o "$build/surface"
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_material_drive.cpp" -o "$build/drive"
"$build/caloric"
"$build/transport"
"$build/surface"
"$build/drive"
printf '%s\n' 'HOST_HELPERS_ONLY: actual caloric/3-D transport/interface helpers passed. OpenFOAM sparse matrix and coupled CUDA execution are separate checks.'
