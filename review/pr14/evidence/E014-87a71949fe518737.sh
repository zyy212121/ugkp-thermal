#!/bin/bash
set -eo pipefail
source /workspace/scratch/98b6339ec81b/chmt_implementation_20261011/native-environment/environment.sh
root=/workspace/scratch/98b6339ec81b/chmt_efficiency_work_20261011
out=/workspace/scratch/98b6339ec81b/chmt_efficiency_refresh_20261011/implementation/chmt_input
export TMPDIR="$out/tmp"
g++ -std=c++17 -O0 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$root/applications/CHMT" -I"$root/common" -I"$out/../build_launch/frontend-chmt/generated" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" "$root/applications/CHMT/tests/test_input_admission_of.C" -L"$FOAM_LIBBIN" -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -o "$out/input_probe"
"$out/input_probe" admitted
