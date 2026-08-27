#!/usr/bin/env bash

test_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
binary="${TMPDIR:-/tmp}/ugkp_full_second_order_gks_host_test"

set -euo pipefail

"${cuda_home}/bin/nvcc" \
    -std=c++17 \
    -O2 \
    "${test_root}/full_second_order_gks_host_test.cu" \
    -o "${binary}"

"${binary}"
rm -f "${binary}"
