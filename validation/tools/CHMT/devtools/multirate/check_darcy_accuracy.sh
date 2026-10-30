#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
build=$(mktemp -d /tmp/chmt-darcy-accuracy.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-O2 -std=c++17 -Wall -Wextra -Werror -pedantic -I"$app" -I"$validation" -I"$app/../../common")
if [[ ${CHMT_SANITIZE:-0} == 1 ]]; then
    flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"$cxx" "${flags[@]}" "$validation/devtools/multirate/test_material_darcy_3d.cpp" \
    "$app/materials/MaterialTransport.C" -o "$build/darcy"
"$build/darcy"
printf '%s\n' 'HOST_COMPONENTS_ONLY: common CHMT material Darcy/reconstruction helpers passed; removed private CUDA gas/material backend is not a verification target.'
