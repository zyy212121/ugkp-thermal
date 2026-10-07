#!/usr/bin/env bash
# Complete available CHMT host gate. No mock CFD, native OF or CUDA claim.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
app=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checks=(
  devtools/check_core.sh
  devtools/check_runtime.sh
  devtools/check_gas_mesh.sh
  devtools/check_material_film.sh
  devtools/multirate/check_sweep_constraints.sh
  devtools/multirate/check_cpu_geometry_state.sh
  gpu/tests/check_gas_window_host.sh
  devtools/multirate/check_darcy_accuracy.sh
  devtools/multirate/check_reaction_accuracy.sh
)
for check in "${checks[@]}"; do
  printf '\n=== %s ===\n' "$check"
  bash "$app/$check"
done
printf '\n%s\n' 'PASS: all available CHMT host gates. Native OF10/CUDA builds and coupled runtime acceptance remain separate required gates.'
