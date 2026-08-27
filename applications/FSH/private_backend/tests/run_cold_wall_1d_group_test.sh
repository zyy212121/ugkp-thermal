#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_89 \
    -I"${root}" \
    "${root}/private_backend/tests/cold_wall_1d_group_test.cu" \
    -o "${build_dir}/cold_wall_1d_group_test"
"${build_dir}/cold_wall_1d_group_test"
