#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
bin=$(mktemp /tmp/chmt-gas-mesh.XXXXXX)
trap 'rm -f "$bin"' EXIT
g++ -std=c++14 -Wall -Wextra -Werror -pedantic -I"$root/applications/CHMT" -I"$root/common" "$root/applications/CHMT/tests/test_gas_mesh.cpp" "$root/applications/CHMT/mesh/Geometry.C" -o "$bin"
"$bin"
