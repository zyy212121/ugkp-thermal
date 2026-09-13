#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
thermal_root=$(realpath -e -- "${root}/..")
case_name=${1:-}
destination=${2:-}

case "${case_name}" in
    MSS7_laminar|MSS7_turbulent_wallModel|MSS7_twoPhase_sparse|MSS7_twoPhase_dense) ;;
    *) echo "Unsupported MSS7 case: ${case_name}" >&2; exit 2 ;;
esac

[[ -n ${destination} ]] || { echo "Missing destination case directory" >&2; exit 2; }
destination=$(realpath -e -- "${destination}")
[[ ${destination} == "${thermal_root}/${case_name}" ]] || {
    echo "Destination does not match ${case_name}: ${destination}" >&2
    exit 2
}

has_time_directory=false
while IFS= read -r -d '' path; do
    name=${path##*/}
    if [[ ${name} =~ ^[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$ ]]; then
        has_time_directory=true
        break
    fi
done < <(find "${destination}" -mindepth 1 -maxdepth 1 -type d -print0)

if ${has_time_directory}; then
    start_from=latestTime
else
    "${root}/restore_initial_state.sh" "${case_name}" "${destination}"
    start_from=startTime
fi

sed -Ei "s/^[[:space:]]*startFrom[[:space:]]+[^;]+;/startFrom       ${start_from};/" "${destination}/system/controlDict"
grep -Eq "^[[:space:]]*startFrom[[:space:]]+${start_from};" "${destination}/system/controlDict"
echo "prepared=${destination} startFrom=${start_from}"
