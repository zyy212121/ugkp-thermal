#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${root}/private_backend/build/host_tests"
mkdir -p "${build_dir}"
/usr/local/cuda/bin/nvcc \
    -std=c++17 \
    -O2 \
    "${root}/private_backend/tests/alumina_properties_host_test.cu" \
    -o "${build_dir}/alumina_properties_host_test"
"${build_dir}/alumina_properties_host_test"
