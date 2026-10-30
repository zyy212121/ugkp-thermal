#!/usr/bin/env bash
# Run from an OpenFOAM 10 shell, after this checkout's Allwmake.
# Requires CUDA, g++, Python 3, NumPy, and a new evidence output directory.
set -eu
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
O=${1:?usage: run_native.sh NEW_OUTPUT_DIRECTORY}
if [[ -e "$O" ]]; then echo "Use a new output directory: $O" >&2; exit 2; fi
mkdir -p "$O"; O=$(cd "$O" && pwd)
: "${WM_PROJECT_DIR:?Source the OpenFOAM 10 environment first}"
CUDA=${CUDA_HOME:-/usr/local/cuda}
ARCH=${CUDA_ARCH:-sm_89}
source "$R/tool/cuda_build_flags.sh"
export UGKP_CUDA_CACHE=0
mkdir "$O/build"
cd "$R"
flags=(-std=c++17 -O2 --fmad=false --expt-relaxed-constexpr -arch="$ARCH" "${ugkp_cuda_flags[@]}" -I. -Icommon -Iapplications/CHMT -Ivalidation/solvers/CHMT)
for unit in constant_wall equilibrium_local_stress wall_history particle_coupling csr_preparation material_sources; do
 ugkp_nvcc "$CUDA/bin/nvcc" "${flags[@]}" "validation/pr33/$unit.cu" -o "$O/build/$unit" > "$O/build/$unit.log" 2>&1
 "$O/build/$unit" > "$O/$unit.log" 2>&1
done
for unit in DiscreteOrdinateRadiation MieOpticalProperties SparseConjugateGradient; do
 ugkp_nvcc "$CUDA/bin/nvcc" "${flags[@]}" -c "common/gpu/$unit.cu" -o "$O/build/$unit.o" > "$O/build/$unit.log" 2>&1
done
link=(-L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" -lcudart)
g++ -std=c++17 -O2 -Icommon -I"$CUDA/include" validation/pr33/radiation_batches.cpp "$O/build/DiscreteOrdinateRadiation.o" "$O/build/MieOpticalProperties.o" "${link[@]}" -o "$O/build/radiation_batches" > "$O/build/radiation_batches.log" 2>&1
"$O/build/radiation_batches" 1024 8 > "$O/radiation_batches.log" 2>&1
for unit in sparse_interfaces specular_reflection; do
 g++ -std=c++17 -O2 -Icommon "validation/pr33/$unit.cpp" "$O/build/SparseConjugateGradient.o" "${link[@]}" -o "$O/build/$unit" > "$O/build/$unit.log" 2>&1
 "$O/build/$unit" > "$O/$unit.log" 2>&1
done
for unit in gas_particles_native wall_profiles_native chmt_output_cadence cht_native; do
 python3 "validation/pr33/$unit.py" --output "$O/$unit" > "$O/$unit.log" 2>&1
done
python3 validation/pr31/chmt_moving_particles_native.py --output "$O/chmt-moving" > "$O/chmt-moving.log" 2>&1
python3 validation/pr31/chmt_particles_native.py --output "$O/chmt-contact" --model reboundContact --impact --end-time .001 --restart-check > "$O/chmt-contact.log" 2>&1
sha256sum bin/gasUGKP bin/gasUGKPSteady bin/CHT bin/CHMT bin/FSHCudaBackend > "$O/applications.sha256"
sha256sum "$O/build"/*.o "$O/build/constant_wall" "$O/build/equilibrium_local_stress" "$O/build/wall_history" "$O/build/particle_coupling" "$O/build/csr_preparation" "$O/build/material_sources" "$O/build/radiation_batches" "$O/build/sparse_interfaces" "$O/build/specular_reflection" > "$O/probes.sha256"
echo 'PASS native PR33 operators and four application consumers'
