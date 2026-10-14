#!/usr/bin/env bash
set -uo pipefail
cd /workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
unset FOAM_USER_APPBIN
export CUDA_HOME=/tmp/pr14_cuda13/nvidia/cu13 NVCC=/tmp/pr14_cuda13/nvidia/cu13/bin/nvcc CHMT_CUDA_ARCH=sm_89
failed=0
for ns in 2 10; do
 export UGKWP_GAS_SPECIES=$ns CHMT_BUILD=/tmp/pr14_compile_checks/chmt_native_ns$ns
 mkdir -p "$CHMT_BUILD"
 date -u +%FT%TZ > "$CHMT_BUILD/start_time.txt"
 sha256sum applications/CHMT/gpu/Backend.cu applications/CHMT/gpu/CourantReductionStorage.cuh common/chemistry/LocalStiffIntegrator.H common/operators/advanceGasChemistryKernel.cuh > "$CHMT_BUILD/start_source_sha256.txt"
 bash -x applications/CHMT/Allwmake > "$CHMT_BUILD/build.log" 2>&1
 rc=$?
 echo "$rc" > "$CHMT_BUILD/exit_code.txt"
 date -u +%FT%TZ > "$CHMT_BUILD/end_time.txt"
 sha256sum -c "$CHMT_BUILD/start_source_sha256.txt" > "$CHMT_BUILD/source_unchanged.log" 2>&1
 printf 'CHMT Ns=%s native compile/link exit=%s\n' "$ns" "$rc"
 tail -6 "$CHMT_BUILD/build.log"
 [ "$rc" = 0 ] || failed=1
done
exit "$failed"
