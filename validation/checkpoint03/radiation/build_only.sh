#!/usr/bin/env bash
# Compile only; run the produced fixtures only at the coordinated checkpoint.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
out=${1:?usage: build_only.sh OUTPUT_DIRECTORY}
mkdir -p "$out";out=$(cd "$out" && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda};arch=${CUDA_ARCH:-sm_89}
source "$root/tool/cuda_build_flags.sh"
flags=(-std=c++17 -O2 --fmad=false -arch="$arch" "${ugkp_cuda_flags[@]}" -I"$root/common")
fixture="$root/validation/checkpoint03/radiation"
ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" "$fixture/ordered_single_scalar.cu" -o "$out/ordered_single_scalar" > "$out/ordered-build.log" 2>&1
for unit in DiscreteOrdinateRadiation MieOpticalProperties; do
 ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" -c "$root/common/gpu/$unit.cu" -o "$out/$unit.o" > "$out/$unit-build.log" 2>&1
done
g++ -std=c++17 -O2 -ffp-contract=off -I"$root/common" -I"$cuda/include" "$fixture/radiation_acceptance.cpp" "$out/DiscreteOrdinateRadiation.o" "$out/MieOpticalProperties.o" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart -o "$out/radiation_acceptance" > "$out/acceptance-build.log" 2>&1
sha256sum "$out/ordered_single_scalar" "$out/radiation_acceptance" "$out/"*.o > "$out/compiled.sha256"
printf 'Compiled fixtures only; no solver, numerical fixture or benchmark was run.\n'
