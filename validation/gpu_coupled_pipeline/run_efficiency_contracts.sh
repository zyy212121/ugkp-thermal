#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$1
baseline=$2
mkdir -p "$out"
nvcc=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
arch=${UGKWP_CUDA_ARCH:-sm_89}
"$nvcc" -std=c++17 -O2 --fmad=false -arch="$arch" -I"$root" "$root/validation/tests/fixtures/shared_operators/empty_moments.cu" -o "$out/empty-moments"
"$out/empty-moments"
for version in baseline candidate; do
 source=$root
 if [[ "$version" = baseline ]]; then source=$baseline; fi
 "$nvcc" -std=c++17 -O2 --fmad=false -arch="$arch" -I"$source/common" -c "$source/common/gpu/PolyhedralGeometry.cu" -o "$out/geometry-$version.o"
 "$nvcc" -std=c++17 -O2 --fmad=false -arch="$arch" -I"$source/common" "$root/validation/gpu_coupled_pipeline/wall_distance_pruning.cpp" "$out/geometry-$version.o" -o "$out/wall-distance-$version"
 "$out/wall-distance-$version" > "$out/wall-distance-$version.log"
done
cmp "$out/wall-distance-baseline.log" "$out/wall-distance-candidate.log"
for test in empty-moments wall-distance-candidate; do
 for mode in memcheck racecheck; do
  "${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool "$mode" --error-exitcode 7 "$out/$test" > "$out/$test-$mode.log" 2>&1
 done
done
