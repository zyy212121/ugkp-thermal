#!/usr/bin/env bash
set -o pipefail
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
unset FOAM_USER_APPBIN
export CUDA_HOME=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/toolchains/cuda
export NVCC="$CUDA_HOME/bin/nvcc"
export CHMT_BUILD=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/chmt/build
export TMPDIR=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/chmt/tmp
export CHMT_CUDA_ARCH=sm_80 UGKWP_GAS_SPECIES=2
cd /workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013
time -p bash applications/CHMT/Allwmake > /workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/chmt/build.log 2>&1
rc=$?
printf '%s\n' "$rc" > /workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/chmt/exit.txt
tail -35 /workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/final-native-build/chmt/build.log
exit "$rc"
