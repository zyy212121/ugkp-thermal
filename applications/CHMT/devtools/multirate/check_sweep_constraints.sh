#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)
bin=$(mktemp /tmp/chmt-sweep-constraints.XXXXXX)
trap 'rm -f "$bin"' EXIT
for test in test_sweep_constraints test_sweep_remainder; do
"${CXX:-g++}" -std=c++14 -Wall -Wextra -Werror -pedantic \
  -I"$root/applications/CHMT" -I"$root/common" \
  "$root/applications/CHMT/devtools/multirate/$test.cpp" \
  "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin"
done
