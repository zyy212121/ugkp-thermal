#!/usr/bin/env bash
# Host source/build/transaction checks; no OpenFOAM or CUDA solver is substituted.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
app=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 "$app/tests/test_multirate_frontend_contract.py"
build=$(mktemp -d /tmp/chmt-multirate-frontend.XXXXXX)
trap 'rm -rf "$build"' EXIT
"${CXX:-g++}" -std=c++14 -Wall -Wextra -Werror -pedantic -I"$app" \
    "$app/tests/test_multirate_orchestration.cpp" "$app/io/MultirateEvolution.C" -o "$build/check"
"$build/check"
