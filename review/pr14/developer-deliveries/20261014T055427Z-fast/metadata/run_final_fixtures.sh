#!/usr/bin/env bash
set -uo pipefail
cd /workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014
export NVCC=/tmp/pr14_cuda13/nvidia/cu13/bin/nvcc
failed=0
for app in CHT FSH; do
 out=/tmp/pr14-cold-$app-frozen
 mkdir -p "$out"
 sha256sum common/chemistry/LocalStiffIntegrator.H tests/fixtures/pr14_fast_cold/initialization.cu > "$out/start_source_sha256.txt"
 date -u +%FT%TZ > "$out/start_time.txt"
 python tests/fixtures/pr14_fast_cold/run.py . "$out" "$app" --nvcc "$NVCC" --compile-only > "$out/driver.log" 2>&1
 rc=$?; echo "$rc" > "$out/exit_code.txt"; date -u +%FT%TZ > "$out/end_time.txt"
 sha256sum -c "$out/start_source_sha256.txt" > "$out/source_unchanged.log" 2>&1
 printf '%s cold fixture CUDA compile/link exit=%s\n' "$app" "$rc"
 tail -6 "$out/driver.log"
 [ "$rc" = 0 ] || failed=1
done
out=/tmp/pr14-chemistry-fixture-frozen
mkdir -p "$out"
sha256sum common/chemistry/LocalStiffIntegrator.H common/operators/advanceGasChemistryKernel.cuh applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh tests/fixtures/chemistry_trial/prepare_chemistry_trial.py tests/fixtures/chemistry_trial/chemistry_trial.cu.in > "$out/start_source_sha256.txt"
date -u +%FT%TZ > "$out/start_time.txt"
python tests/fixtures/chemistry_trial/prepare_chemistry_trial.py . "$out" --nvcc "$NVCC" --arch sm_89 > "$out/driver.log" 2>&1
rc=$?; echo "$rc" > "$out/exit_code.txt"; date -u +%FT%TZ > "$out/end_time.txt"
sha256sum -c "$out/start_source_sha256.txt" > "$out/source_unchanged.log" 2>&1
printf 'Chemistry owner fixture CUDA compile/link exit=%s\n' "$rc"
tail -6 "$out/driver.log"
[ "$rc" = 0 ] || failed=1
exit "$failed"
