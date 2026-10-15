"""Compatibility entry point; validation uses the common S1/L2 pipeline."""
from pathlib import Path as _ValidationPath
import sys as _validation_sys
_validation_sys.path.insert(0, str(next(p.parent for p in _ValidationPath(__file__).resolve().parents if p.name == "validation")))
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import sys
raise SystemExit(subprocess.call([sys.executable, str((VALIDATION_ROOT / 'tools/tests/fixtures/thermal_workers/s1_pipeline.py')), *sys.argv[1:]]))
