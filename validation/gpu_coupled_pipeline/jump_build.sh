#!/usr/bin/env bash
# Source OpenFOAM 10 first. Compile every linked CUDA unit from source.
set -euo pipefail
if [[ $# != 4 ]]; then echo "usage: $0 FIXED_REPOSITORY FROZEN_88BC FROZEN_2228479A OUTPUT" >&2; exit 2; fi
R=$(realpath "$1"); B=$(realpath "$2"); F=$(realpath "$3"); O=$(realpath -m "$4")
mkdir -p "$O/build"
for unit in ImplicitDiffusion ThermalFieldOperators; do
 /usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_89 -c "$R/common/gpu/$unit.cu" -o "$O/build/$unit.o" > "$O/build/$unit.log" 2>&1
done
flags=(-std=c++17 -O2 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude")
libs=(-L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart)
for kind in before after; do
 tree=$R
 if [[ $kind == before ]]; then tree=$F; fi
 /usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_89 -I"$R/common/gpu" -c "$tree/common/gpu/SparseConjugateGradient.cu" -o "$O/build/sparse-$kind.o" > "$O/build/sparse-$kind.log" 2>&1
 g++ "${flags[@]}" -I"$tree/common" -I"$R/common" "$R/validation/gpu_coupled_pipeline/solid_jump_foam.cpp" "$O/build/sparse-$kind.o" "$O/build/ImplicitDiffusion.o" "$O/build/ThermalFieldOperators.o" "${libs[@]}" -o "$O/build/jump-$kind" > "$O/build/jump-$kind.log" 2>&1
done
g++ -std=c++17 -O2 -I"$R/common" "$R/validation/tests/sparse_conjugate_gradient.cpp" "$O/build/sparse-after.o" -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -o "$O/build/sparse-unit" > "$O/build/sparse-unit-build.log" 2>&1
"$O/build/sparse-unit" > "$O/build/sparse-unit.log" 2>&1
sha256sum "$O/build"/*.o "$O/build/jump-before" "$O/build/jump-after" "$O/build/sparse-unit" > "$O/build/binaries.sha256"/usr/local/cuda/bin/nvcc -std=c++17 -O2 --fmad=false -arch=sm_89 -c "$R/common/gpu/SolidThermalProperties.cu" -o "$O/build/SolidThermalProperties.o" > "$O/build/SolidThermalProperties.log" 2>&1
flags=(-std=c++17 -O2 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude" -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude")
libs=(-L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream)
for kind in baseline candidate; do
 tree=$R;extra=("$O/build/sparse-after.o" "$O/build/ImplicitDiffusion.o" "$O/build/ThermalFieldOperators.o" "$O/build/SolidThermalProperties.o" -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart)
 if [[ $kind == baseline ]]; then tree=$B;extra=();fi
 thermal=$tree/applications/CHT/thermal
 g++ "${flags[@]}" -I"$thermal" "$R/validation/gpu_coupled_pipeline/solid_jump_owner.cpp" "$thermal/GpuSolidThermalCoupler.C" "$thermal/GpuSolidThermalProperties.C" "$thermal/GpuThermalExchangeState.C" "${extra[@]}" "${libs[@]}" -o "$O/build/owner-$kind" > "$O/build/owner-$kind.log" 2>&1
done
sha256sum "$O/build/owner-baseline" "$O/build/owner-candidate" > "$O/build/owner-binaries.sha256"