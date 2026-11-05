#!/usr/bin/env bash
R=$(cd "$(dirname "$0")/.." && pwd)
unset GAS_UGKP_BUILD GAS_UGKP_STEADY_BUILD CHMT_BUILD NVCC CHMT_CUDA_ARCH UGKWP_CUDA_FAST_COMPILE
/usr/bin/time -v bash "$R/tool/build_local.sh" > "$R/build-evidence/allwmake-local.log" 2>&1
status=$?
printf '%s\n' "$status" > "$R/build-evidence/allwmake-local.exit"
exit "$status"
