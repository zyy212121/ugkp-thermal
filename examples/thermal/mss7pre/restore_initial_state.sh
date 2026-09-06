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

source_time="${root}/checkpoints/${case_name}/1"
[[ -d ${source_time} ]] || { echo "Missing pre-developed checkpoint: ${source_time}" >&2; exit 2; }
rm -rf -- "${destination}/1"
cp -a -- "${source_time}" "${destination}/1"
echo "restored=${destination}/1 source=${source_time}"
