"""Geometric reflection must not dissipate energy or trap an axis-crossing parcel."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_geometric_boundaries_preserve_particles_and_elastic_energy(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/fixtures/particle_tracking/geometry_behavior.py"),
         str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
