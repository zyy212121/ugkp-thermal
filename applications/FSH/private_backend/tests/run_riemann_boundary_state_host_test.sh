#!/usr/bin/env bash

test_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
binary="${TMPDIR:-/tmp}/ugkp_riemann_boundary_state_host_test"

set -euo pipefail
trap 'rm -f "${binary}"' EXIT

"${cuda_home}/bin/nvcc" \
    -std=c++17 \
    -O2 \
    -Werror cross-execution-space-call \
    "${test_root}/riemann_boundary_state_host_test.cu" \
    -o "${binary}"

"${binary}"
