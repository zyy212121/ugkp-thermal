#!/usr/bin/env bash
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_dir="$(cd "${script_dir}/../.." && pwd)"

if [ -z "${WM_PROJECT_DIR:-}" ]; then echo "OpenFOAM environment is not loaded" >&2; exit 1; fi
set -eu
python3 "${script_dir}/prepare_initial_state.py" create
(
    cd "${script_dir}/preconditioning_case"
    gasUGKP > log.preconditioning 2>&1
)

last_time="$(foamListTimes -case "${script_dir}/preconditioning_case" -latestTime)"
python3 "${script_dir}/prepare_initial_state.py" install --source-time "${last_time}"
postProcess -case "${case_dir}" -region graphite -time 1.5 -func writeCellCentres > "${script_dir}/log.writeGraphiteCentres" 2>&1
python3 "${script_dir}/initialise_graphite_temperature.py"
python3 "${script_dir}/finalise_thermal_restart.py"
