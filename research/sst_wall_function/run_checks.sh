#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export OPENBLAS_NUM_THREADS=1
cd "$ROOT"
python3 "$HERE/resolved_sst_reference.py"
python3 "$HERE/run_owner32.py"
python3 "$HERE/compare_couette.py"
python3 "$HERE/run_blowing.py"
python3 "$HERE/verify_results.py"
