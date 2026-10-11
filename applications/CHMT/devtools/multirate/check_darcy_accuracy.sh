#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-darcy-accuracy.XXXXXX)
trap 'rm -rf "$build"' EXIT
cxx=${CXX:-g++}
flags=(-O2 -std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" -I"$app/../../common")
if [[ ${CHMT_SANITIZE:-0} == 1 ]]; then
    flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"$cxx" "${flags[@]}" "$app/devtools/multirate/test_material_darcy_3d.cpp" \
    "$app/materials/MaterialTransport.C" -o "$build/darcy"
"$build/darcy"
printf '%s\n' 'HOST_COMPONENTS_ONLY: common CHMT material Darcy/reconstruction helpers passed; removed private CUDA gas/material backend is not a verification target.'
