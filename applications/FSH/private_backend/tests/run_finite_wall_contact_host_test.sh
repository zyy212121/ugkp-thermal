#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT
g++ -std=c++17 -O2 -x c++ -I"${root}" \
    "${root}/private_backend/tests/finite_wall_contact_host_test.cu" \
    -o "${build_dir}/finite_wall_contact_host_test"
"${build_dir}/finite_wall_contact_host_test"
