#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for target in "${case_dir}/.validation-build" "${case_dir}/.validation-bin"; do
    resolved_parent="$(cd "$(dirname "${target}")" && pwd)"
    [[ "${resolved_parent}" == "${case_dir}" ]] || {
        echo "Refusing to clean path outside case: ${target}" >&2
        exit 8
    }
    rm -rf -- "${target}"
done
