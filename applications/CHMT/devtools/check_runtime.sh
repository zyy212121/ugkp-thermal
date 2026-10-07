#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/chmt-runtime.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
cxx=${CXX:-g++}
flags=(-std=c++14 -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Werror -pedantic -Iapplications/CHMT -Icommon)
"$cxx" "${flags[@]}" applications/CHMT/tests/test_transaction.cpp applications/CHMT/mesh/Geometry.C -o "$scratch/transaction"
"$scratch/transaction"
"$cxx" "${flags[@]}" applications/CHMT/tests/test_restart.cpp applications/CHMT/restart/Checkpoint.C applications/CHMT/mesh/Geometry.C -Wl,--wrap=fsync -o "$scratch/restart"
"$scratch/restart"
"$cxx" "${flags[@]}" applications/CHMT/tests/test_particles.cpp -o "$scratch/particles"
"$scratch/particles"
for phase in dryout birth; do
  "$cxx" "${flags[@]}" "applications/CHMT/tests/test_phase_${phase}_transaction.cpp" \
    applications/CHMT/io/MultirateEvolution.C applications/CHMT/film/CpuFilmDriver.C -o "$scratch/phase_$phase"
  "$scratch/phase_$phase"
done
# The historical test_runtime_sources.py was never present in this checkout.
# Exercise the maintained contracts and real macro transaction helper instead.
bash applications/CHMT/devtools/check_interval_contracts.sh
bash applications/CHMT/devtools/check_multirate_frontend.sh
printf '%s\n' 'HOST_ONLY: packet/resource/restart/particle and macro contracts passed; CUDA/native CFD are separate gates.'
