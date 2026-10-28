#!/usr/bin/env bash
# Correctness-only reproduction; the output directory must be new.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$1
mkdir "$out"
out=$(cd "$out" && pwd)
cd "$root"
git rev-parse HEAD > "$out/source-commit.txt"
git diff --binary > "$out/source-working-tree.patch"
bash tool/build_local.sh > "$out/build.log" 2>&1
python3 validation/composition_transport/pr_wall_regression.py "$out/host" > "$out/host.log"
python3 validation/composition_transport/pr_wall_regression.py "$out/cuda" --cuda > "$out/cuda.log"
python3 -m pytest -q validation/tests/gas_transport/test_mixture_wall_function.py \
    validation/tests/gas_transport/test_review_sst_wall_audits.py \
    validation/composition_transport/test_configuration.py \
    > "$out/related-tests.log" 2>&1
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    "$out/cuda/probe" > "$out/memcheck.log" 2>&1
sha256sum bin/gasUGKP bin/gasUGKPSteady bin/FSH bin/FSHCudaBackend bin/CHT bin/CHMT \
    "$out/host/probe" "$out/cuda/probe" > "$out/binaries.sha256"
