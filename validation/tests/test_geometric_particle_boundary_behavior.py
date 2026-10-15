"""Geometric reflection must not dissipate energy or trap an axis-crossing parcel."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import sys

ROOT = SOURCE_ROOT

def test_geometric_boundaries_preserve_particles_and_elastic_energy(tmp_path):
    result = subprocess.run(
        [sys.executable, str((VALIDATION_ROOT / 'tools/tests/fixtures/particle_tracking/geometry_behavior.py')),
         str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
