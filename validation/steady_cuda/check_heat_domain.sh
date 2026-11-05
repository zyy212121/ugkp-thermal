#!/usr/bin/env bash
# Run the production paired CUDA kernel against an independent analytic root.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?usage: check_heat_domain.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
"$nvcc" -std=c++17 --expt-relaxed-constexpr -O2 --fmad=true -arch="${UGKWP_CUDA_ARCH:-sm_89}" \
  -I"$root" -I"$root/common" -I"$root/common/gasNumerics" \
  "$root/validation/tests/gas_transport/steady_mixture_heat_domain.cu" -o "$out/steady_mixture_heat_domain"
for args in '0.00002 -0.22 10000' '100 -0.22 10000' '0.00002 0 10000' '0.00002 -0.22 2500'; do
  echo "case $args"
  "$out/steady_mixture_heat_domain" $args
done
