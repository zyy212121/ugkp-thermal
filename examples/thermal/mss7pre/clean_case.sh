#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
thermal_root=$(realpath -e -- "${root}/..")
case_name=${1:-}
case_dir="${thermal_root}/${case_name}"

case "${case_name}" in
    MSS7_laminar|MSS7_turbulent_wallModel|MSS7_twoPhase_sparse|MSS7_twoPhase_dense) ;;
    *) echo "Unsupported MSS7 case: ${case_name}" >&2; exit 2 ;;
esac

[[ -f "${case_dir}/system/controlDict" ]] || { echo "Invalid case: ${case_dir}" >&2; exit 2; }
for proc_cwd in /proc/[0-9]*/cwd; do
    running_cwd=$(readlink -f -- "${proc_cwd}" 2>/dev/null || true)
    if [[ ${running_cwd} == "${case_dir}" || ${running_cwd} == "${case_dir}/"* ]]; then
        proc_pid=${proc_cwd%/cwd}
        proc_comm=$(cat "${proc_pid}/comm" 2>/dev/null || true)
        [[ ${proc_comm} == bash || ${proc_comm} == sh || ${proc_comm} == zsh || ${proc_comm} == fish ]] && continue
        echo "Refusing to clean active case ${case_dir} (process cwd ${proc_cwd%/cwd})." >&2
        exit 4
    fi
done

while IFS= read -r -d '' path; do
    rm -rf -- "${path}"
done < <(
    find "${case_dir}" -mindepth 1 -maxdepth 1 -type d -print0 |
    while IFS= read -r -d '' path; do
        name=${path##*/}
        [[ ${name} =~ ^[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$ ]] && printf '%s\0' "${path}"
    done
)

rm -rf -- "${case_dir}/postProcessing" "${case_dir}/VTK" "${case_dir}/dynamicCode"
find "${case_dir}" -mindepth 1 -maxdepth 1 -type f \
    \( -name 'log' -o -name 'log.*' -o -name '*.log' -o -name 'case.foam' -o -name '*.OpenFOAM' -o -name 'core' -o -name 'core.*' \) \
    -delete
rm -f -- "${case_dir}/assets/radiation/alumina_mieTable.dat"
echo "cleaned=${case_dir} initialCheckpoint=external"
