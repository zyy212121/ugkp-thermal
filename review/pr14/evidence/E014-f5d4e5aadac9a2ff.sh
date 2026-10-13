#!/usr/bin/env bash
set -eo pipefail
ROOT=${1:-/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013}
AUDIT=${2:-/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/thermal-terminal}
source /workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh
mkdir -p "$AUDIT/native-compile-only/Make"
: > "$AUDIT/native-compile-only/Make/files"
for owner in GpuSolidThermalCoupler GpuSolidThermalProperties GpuSolidPatchMapper GpuThermalExchangeState GpuParticleRadiationCoupler MieDoRadiationSolver MieTable; do
  printf '%s/applications/CHT/thermal/%s.C\n' "$ROOT" "$owner" >> "$AUDIT/native-compile-only/Make/files"
done
printf 'LIB = %s/native-compile-only/libThermalCompileOnly\n' "$AUDIT" >> "$AUDIT/native-compile-only/Make/files"
cat > "$AUDIT/native-compile-only/Make/options" <<EOF
EXE_INC = -std=c++17 -DUGKWP_USE_CUDA -I$ROOT/applications/CHT/thermal -I$ROOT/applications/CHT/gpu -I$ROOT/common -I\$(LIB_SRC)/finiteVolume/lnInclude -I\$(LIB_SRC)/meshTools/lnInclude
LIB_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM
EOF
cd "$AUDIT/native-compile-only"
wmake libso
# Compile/shared-library link only. No executable or numerical test is run.
