#!/usr/bin/env bash


set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
root="$source_root"
app="$root/applications/CHMT"
validation="$root/validation/solvers/CHMT"
runners="$root/validation/tools/CHMT"
bash "$runners/devtools/multirate/check_cpu_material.sh"
bash "$runners/devtools/multirate/check_cpu_film.sh"
