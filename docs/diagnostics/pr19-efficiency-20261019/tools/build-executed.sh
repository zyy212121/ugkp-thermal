#!/bin/bash
source /opt/openfoam10/etc/bashrc
set -eo pipefail
R=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-efficiency-20261019
W=/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/perf-efficiency-20261019
mkdir -p "$W/install/bin" "$W/install/lib" "$W/tmp" "$W/cache"
export FOAM_USER_APPBIN="$W/install/bin" FOAM_USER_LIBBIN="$W/install/lib"
export CUDA_HOME=/usr/local/cuda-13.1 UGKWP_CUDA_ARCH=sm_89 UGKWP_GAS_SPECIES=2
export CHMT_CUDA_ARCH=sm_89
export GAS_UGKP_FMAD=1 CUDA_CACHE_PATH="$W/cache/cuda" XDG_CACHE_HOME="$W/cache" TMPDIR="$W/tmp"
export PATH="$FOAM_USER_APPBIN:$FOAM_APPBIN:$CUDA_HOME/bin:/usr/bin:/bin:$WM_PROJECT_DIR/wmake"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN/dummy:$CUDA_HOME/lib64"
unset GAS_UGKP_CUDA_BACKEND UGKP_GAS_BUDGET_OUTPUT
cd "$R"
date -Iseconds > "$W/build-start.txt"
set +e
bash ./Allwmake > "$W/full-build.log" 2>&1
status=$?
printf '%s\n' "$status" > "$W/full-build.exit"
date -Iseconds > "$W/build-end.txt"
tail -30 "$W/full-build.log"
exit "$status"
