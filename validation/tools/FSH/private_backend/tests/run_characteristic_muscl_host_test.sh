#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
test_dir="${source_root}/validation/solvers/FSH/private_backend/tests"
nvcc -std=c++17 -O2 -x cu \
    "${test_dir}/characteristic_muscl_host_test.cu" \
    -o "${test_dir}/characteristic_muscl_host_test"
"${test_dir}/characteristic_muscl_host_test"
