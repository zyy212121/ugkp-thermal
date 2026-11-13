#!/usr/bin/env bash
# Compilation only. Does not launch any fixture or benchmark.
# The parent checkpoint runner owns the one active nvcc slot.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
out=${1:?usage: build_only.sh OUTPUT_DIRECTORY}
mkdir -p "$out";out=$(cd "$out" && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda};arch=${CUDA_ARCH:-sm_89}
source "$root/tool/cuda_build_flags.sh"
export UGKP_CUDA_CACHE=0
flags=(-std=c++17 -O2 --fmad=false --expt-relaxed-constexpr -arch="$arch" "${ugkp_cuda_flags[@]}" -I"$root/common")
fixture="$root/validation/checkpoint02/radiation"
ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" "$fixture/ordered_radiation.cu" -o "$out/ordered_radiation" > "$out/ordered_radiation-build.log" 2>&1
ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" -c "$root/common/gpu/DiscreteOrdinateRadiation.cu" -o "$out/DiscreteOrdinateRadiation.o" > "$out/radiation-build.log" 2>&1
ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" -include "$fixture/recovery_counter.cuh" -c "$root/common/gpu/DiscreteOrdinateRadiation.cu" -o "$out/DiscreteOrdinateRadiation-counted.o" > "$out/radiation-counted-build.log" 2>&1
ugkp_nvcc "$cuda/bin/nvcc" "${flags[@]}" -c "$root/common/gpu/MieOpticalProperties.cu" -o "$out/MieOpticalProperties.o" > "$out/mie-build.log" 2>&1
link=(-L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lcudart)
g++ -std=c++17 -O2 -ffp-contract=off -I"$root/common" -I"$cuda/include" "$fixture/radiation_acceptance.cpp" "$out/DiscreteOrdinateRadiation-counted.o" "$out/MieOpticalProperties.o" "${link[@]}" -o "$out/radiation_acceptance" > "$out/acceptance-build.log" 2>&1
sha256sum "$out/ordered_radiation" "$out/radiation_acceptance" "$out/"*.o > "$out/compiled.sha256"
printf 'Compiled fixtures only. No numerical execution was performed.\n'
