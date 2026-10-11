#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/../.." && pwd)
build=$(mktemp -d /tmp/chmt-reaction-accuracy.XXXXXX)
trap 'rm -rf "$build"' EXIT
flags=(-O2 -std=c++14 -Wall -Wextra -Werror -pedantic)
if [[ ${SANITIZE:-0} == 1 ]]; then flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
${CXX:-g++} "${flags[@]}" -I"$app" -I"$app/../../common" "$app/devtools/multirate/test_reaction_accuracy.cpp" "$app/materials/MaterialTransport.C" -o "$build/test"
"$build/test"
