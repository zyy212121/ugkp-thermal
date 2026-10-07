#!/usr/bin/env bash
# Current production components, replacing an absent historical test target.
# These are host helper/candidate tests, not native OpenFOAM conduction.
set -euo pipefail
app=$(cd "$(dirname "$0")/.." && pwd)
bash "$app/devtools/multirate/check_cpu_material.sh"
bash "$app/devtools/multirate/check_cpu_film.sh"
