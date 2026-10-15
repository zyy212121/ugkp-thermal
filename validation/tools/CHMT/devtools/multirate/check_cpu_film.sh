#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
build=$(mktemp -d /tmp/chmt-cpu-film.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-O2 -std=c++14 -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Werror -pedantic -I"$app" -I"$validation" -I"$root/common")
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_cpu_film.cpp" "$app/film/CpuFilmDriver.C" -o "$build/film"
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_cpu_film_curved.cpp" "$app/film/CpuFilmDriver.C" "$app/mesh/Geometry.C" -o "$build/curved"
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_cpu_phase_lifecycle.cpp" \
  "$app/film/CpuFilmDriver.C" "$app/ablation/CpuSurfaceInterface.C" \
  "$app/materials/MaterialTransport.C" "$app/mesh/Geometry.C" -o "$build/lifecycle"
"$build/film"
"$build/curved"
"$build/lifecycle"
printf '%s\n' 'HOST_COMPONENTS_ONLY: native OpenFOAM/CUDA interval-driver execution is a separate required check.'
