#!/usr/bin/env bash
set -euo pipefail
test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nvcc -std=c++17 -O2 -x cu \
    "${test_dir}/characteristic_muscl_host_test.cu" \
    -o "${test_dir}/characteristic_muscl_host_test"
"${test_dir}/characteristic_muscl_host_test"
