#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
root="${source_root}/applications/FSH"
build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_89 \
    -I"${root}" -I"${root}/../../common" \
    "${source_root}/validation/solvers/FSH/private_backend/tests/cold_wall_1d_group_test.cu" \
    -o "${build_dir}/cold_wall_1d_group_test"
"${build_dir}/cold_wall_1d_group_test"
