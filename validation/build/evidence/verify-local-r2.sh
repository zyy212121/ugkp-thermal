#!/usr/bin/env bash
R=$(cd "$(dirname "$0")/.." && pwd)
source "$R/env.sh"
set -eo pipefail
python3 "$R/validation/build/check_local.py" --bin-dir "$R/bin" --output "$R/../case/local-verification-r2" > "$R/build-evidence/native-verification-r2.log" 2>&1
