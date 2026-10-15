#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="${source_root}/applications/FSH"
build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT
g++ -std=c++17 -O2 -x c++ -I"${root}" -I"${root}/../../common" \
    "${source_root}/validation/solvers/FSH/private_backend/tests/finite_wall_contact_host_test.cu" \
    -o "${build_dir}/finite_wall_contact_host_test"
"${build_dir}/finite_wall_contact_host_test"
