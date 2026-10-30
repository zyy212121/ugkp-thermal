#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
build=$(mktemp -d /tmp/chmt-reaction-accuracy.XXXXXX)
trap 'rm -rf "$build"' EXIT
flags=(-O2 -std=c++17 -Wall -Wextra -Werror -pedantic)
if [[ ${SANITIZE:-0} == 1 ]]; then flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
${CXX:-g++} "${flags[@]}" -I"$app" -I"$validation" -I"$app/../../common" "$validation/devtools/multirate/test_reaction_accuracy.cpp" "$app/materials/MaterialTransport.C" -o "$build/test"
"$build/test"
