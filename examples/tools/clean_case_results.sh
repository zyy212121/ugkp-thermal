#!/usr/bin/env bash

set -euo pipefail

usage()
{
    echo "Usage: $0 [--check] CASE_DIR" >&2
    exit 2
}

check_only=false
if [[ ${1:-} == --check ]]; then
    check_only=true
    shift
fi
[[ $# -eq 1 ]] || usage

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
examples_root=$(realpath -e -- "${script_dir}/..")
case_dir=$(realpath -e -- "$1")

case "${case_dir}/" in
    "${examples_root}/"*) ;;
    *)
        echo "Refusing to clean a path outside ${examples_root}: ${case_dir}" >&2
        exit 2
        ;;
esac
[[ -f "${case_dir}/system/controlDict" ]] || {
    echo "Not an OpenFOAM case: ${case_dir}" >&2
    exit 2
}

for proc_cwd in /proc/[0-9]*/cwd; do
    running_cwd=$(readlink -f -- "${proc_cwd}" 2>/dev/null || true)
    if [[ ${running_cwd} == "${case_dir}" || ${running_cwd} == "${case_dir}/"* ]]; then
        proc_pid=${proc_cwd%/cwd}
        proc_cmdline=$(tr '\0' ' ' < "${proc_pid}/cmdline" 2>/dev/null || true)
        proc_comm=$(cat "${proc_pid}/comm" 2>/dev/null || true)
        if [[ ${proc_comm} == bash && ${proc_cmdline} == *"shellIntegration-bash.sh"* ]]; then
            continue
        fi
        echo "Refusing to clean active case ${case_dir} (process cwd ${proc_cwd%/cwd})." >&2
        exit 4
    fi
done

mapfile -t time_dirs < <(
    find "${case_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
    awk '$0 ~ /^[-+]?[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/ {print}' |
    sort -g
)
(( ${#time_dirs[@]} >= 1 )) || {
    echo "No protected numeric checkpoint found in ${case_dir}." >&2
    exit 2
}

protected_time=${time_dirs[0]}
if ${check_only}; then
    echo "clean-check case=${case_dir} protectedTime=${protected_time}"
    exit 0
fi

safe_remove()
{
    local target=$1
    [[ -e ${target} || -L ${target} ]] || return 0
    local parent
    parent=$(realpath -e -- "$(dirname "${target}")")
    [[ ${parent} == "${case_dir}" ]] || {
        echo "Refusing target outside case root: ${target}" >&2
        exit 2
    }
    rm -rf -- "${target}"
}

for ((i=1; i<${#time_dirs[@]}; ++i)); do
    safe_remove "${case_dir}/${time_dirs[i]}"
done

for name in postProcessing VTK dynamicCode .validation-build .validation-bin; do
    safe_remove "${case_dir}/${name}"
done

while IFS= read -r -d '' generated_dir; do
    safe_remove "${generated_dir}"
done < <(
    find "${case_dir}" -mindepth 1 -maxdepth 1 -type d \
        \( -name 'processor[0-9]*' -o -name 'processors[0-9]*' \) -print0
)

while IFS= read -r -d '' generated_file; do
    safe_remove "${generated_file}"
done < <(
    find "${case_dir}" -mindepth 1 -maxdepth 1 -type f \
        \( -name 'log' -o -name 'log.*' -o -name '*.log' -o \
           -name 'case.foam' -o -name '*.OpenFOAM' -o \
           -name 'core' -o -name 'core.*' \) -print0
)

echo "cleaned=${case_dir} protectedTime=${protected_time}"
