#!/usr/bin/env bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
# Source an official OF10 environment or set OF10_ENV to an installation wrapper.
if [ -n "${OF10_ENV:-}" ]; then source "$OF10_ENV" || exit $?; fi
: "${WM_PROJECT_DIR:?Source OpenFOAM10 or set OF10_ENV}"
export DRY_CORE_ROOT="${DRY_CORE_ROOT:-$(cd "$HERE/../.." && pwd)}"
export DRY_VALIDATION_BIN="$HERE/bin"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$HERE/bin" "$HERE/results"
(cd "$HERE/driver" && wmake) > "$HERE/results/build.log" 2>&1
python3 "$HERE/run_cases.py" > "$HERE/results/run_summary.log"
python3 "$HERE/run_extended.py" > "$HERE/results/extended_run.log"
python3 "$HERE/independent_coupled_reference.py" > "$HERE/results/independent_run.log"
python3 - "$HERE" <<'PY'
import sys,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from run_cases import case
from independent_coupled_reference import reference
(root/'results/surface_corrected.json').write_text(json.dumps([case('surface',n,.01,1) for n in [10,20,40]],indent=2))
(root/'results/mixture_summary.json').write_text(json.dumps([case('mixture',n,.01,1) for n in [20,40,80,160,320]],indent=2))
(root/'results/zero_support_summary.json').write_text(json.dumps([case('zeroSupport',n,.0001,.01) for n in [8,16]],indent=2))
reference(320);reference(640)
(root/'results/threeD_summary.json').write_text(json.dumps([case(mode,n,.0001,.1) for mode in ['conduction3d','diffusion3d'] for n in [8,16]],indent=2))
PY
python3 "$HERE/assess.py" > "$HERE/results/assessment.log"
