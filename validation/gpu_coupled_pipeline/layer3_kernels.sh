#!/usr/bin/env bash
# Source OpenFOAM10 first. All outputs belong in a fresh evidence directory.
set -euo pipefail
if [[ $# != 3 ]]; then echo "usage: $0 REPOSITORY FROZEN_88BC_CHECKOUT OUTPUT" >&2; exit 2; fi
root=$(realpath "$1"); baseline=$(realpath "$2"); out=$(realpath -m "$3")
mkdir "$out"
cuda=${CUDA_HOME:-/usr/local/cuda}; arch=${UGKWP_CUDA_ARCH:-sm_89}
flags=(-std=c++17 -O2 -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -I"$root/common"
 -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/finiteVolume/lnInclude"
 -I"$WM_PROJECT_DIR/src/meshTools/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude")
libs=(-L"$FOAM_LIBBIN" -lfiniteVolume -lmeshTools -lOpenFOAM -L"$FOAM_LIBBIN/$FOAM_MPI" -lPstream
 -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart)
for unit in SparseConjugateGradient SolidThermalProperties ImplicitDiffusion ThermalFieldOperators; do
 "$cuda/bin/nvcc" -std=c++17 -O2 --fmad=false -arch="$arch" -c "$root/common/gpu/$unit.cu" -o "$out/$unit.o" > "$out/build-$unit.log" 2>&1
done
g++ -std=c++17 -O2 -I"$root/common" "$root/validation/tests/sparse_conjugate_gradient.cpp" "$out/SparseConjugateGradient.o" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart -o "$out/sparse-test" > "$out/build-sparse-test.log" 2>&1
"$out/sparse-test" > "$out/sparse-test.log" 2>&1
thermal="$baseline/applications/CHT/thermal"
g++ "${flags[@]}" -I"$thermal" "$root/validation/gpu_coupled_pipeline/solid_properties_cuda.cpp" "$thermal/GpuSolidThermalProperties.C" "$out/SolidThermalProperties.o" "${libs[@]}" -o "$out/property-test" > "$out/build-property-test.log" 2>&1
"$out/property-test" > "$out/property-test.log" 2>&1
for species in 2 3; do
 "$cuda/bin/nvcc" -std=c++17 -O2 --fmad=false --expt-relaxed-constexpr -arch="$arch" -DUGKWP_GAS_SPECIES=$species -I"$root/applications/CHMT" -I"$root/common" -c "$root/applications/CHMT/gpu/MaterialThermalBackend.cu" -o "$out/MaterialThermalBackend-$species.o" > "$out/build-material-$species.log" 2>&1
 g++ -std=c++17 -O2 -DUGKWP_GAS_SPECIES=$species -I"$root/applications/CHMT" -I"$root/common" -I"$root/validation/solvers/CHMT" "$root/validation/solvers/CHMT/tests/material_thermal_cuda.cpp" "$out/MaterialThermalBackend-$species.o" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart -o "$out/material-test-$species" >> "$out/build-material-$species.log" 2>&1
 "$out/material-test-$species" > "$out/material-test-$species.log" 2>&1
done
for unit in solid_sparse thermal_fields; do
 g++ "${flags[@]}" "$root/validation/gpu_coupled_pipeline/${unit}_foam.cpp" "$out/SparseConjugateGradient.o" "$out/ImplicitDiffusion.o" "$out/ThermalFieldOperators.o" "${libs[@]}" -o "$out/$unit" > "$out/build-$unit.log" 2>&1
done
sha256sum "$out"/*.o "$out/sparse-test" "$out/property-test" "$out/material-test-2" "$out/material-test-3" "$out/solid_sparse" "$out/thermal_fields" > "$out/binaries.sha256"
printf '%s\n' 'PASS standalone sparse and property oracles; mesh operators are built for layer3_mesh.py.'
