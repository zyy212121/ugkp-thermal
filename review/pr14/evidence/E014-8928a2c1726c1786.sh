#!/bin/bash
# Compilation only: intentionally contains no generated-binary/test execution.
saved_ld=${LD_LIBRARY_PATH:-}
set +e
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
set -euo pipefail
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH${saved_ld:+:$saved_ld}"
root=/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013
out=/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/frontend-terminal/v2
export TMPDIR="$out/tmp"
mkdir -p "$out/compile" "$out/logs" "$out/tmp"
cat > "$out/compile/frontend.C" <<'CPP'
#include "gpu/GasWallMath.H"
#include "configuration/CouplingInput.H"
#include "io/CoupledOutput.H"
#include "gpu/StateValidation.H"
#include "mesh/EmptyExtrusion.H"
#include "BoundaryLayerInput.H"
#include "SharedGasModelInput.H"
bool compiledEmptyTerminal(const chmt::HostMesh& mesh,chmt::empty_detail::Extrusion& out,std::string& error){return chmt::empty_detail::build<true>(mesh,out,error);}
bool compiledEmptyStrict(const chmt::HostMesh& mesh,chmt::empty_detail::Extrusion& out,std::string& error){return chmt::empty_detail::build<false>(mesh,out,error);}
CPP
flags=(-I"/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/delivery/first-round/first-round-reproduction/helpers" -std=c++17 -O0 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$root/applications/CHMT" -I"$root/common" -I"$root/applications/gasUGKP/gpu" -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude")
for source in "$out/compile/frontend.C" "$root/applications/CHMT/mesh/Geometry.C" "$root/applications/CHMT/materials/MaterialTransport.C" "$root/applications/CHMT/tests/test_input_admission_of.C";do
 name=$(basename "$source" .C)
 printf '%q ' g++ "${flags[@]}" -c "$source" -o "$out/compile/$name.o";printf '\n'
 g++ "${flags[@]}" -c "$source" -o "$out/compile/$name.o" >"$out/logs/compile-$name.log" 2>&1
done
