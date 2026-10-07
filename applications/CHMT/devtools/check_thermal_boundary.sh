#!/usr/bin/env bash
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
build=$(mktemp -d /tmp/chmt-thermal-boundary.XXXXXX)
trap 'rm -rf "$build"' EXIT
"${CXX:-g++}" -std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" -I"$app/../../common" \
 "$app/tests/test_thermal_boundary.cpp" -o "$build/test"
"$build/test"
printf '%s\n' 'HOST_MATH: production wall sampling/flux/EOS policy helpers passed; native field import and CUDA execution are separate gates.'
