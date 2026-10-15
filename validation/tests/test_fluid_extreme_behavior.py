from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT = SOURCE_ROOT

def test_unsorted_pressure_preserves_limited_flux_and_moments(tmp_path):
    """Exercise the reachable unsorted CUDA pipeline, including its shared dependencies."""
    if not Path('/usr/local/cuda/bin/nvcc').is_file():
        pytest.skip('CUDA compiler required')
    result = subprocess.run(['python3', str((VALIDATION_ROOT / 'tools/tests/fixtures/pressure/limited_flux_balance.py')),
        str(ROOT), str(tmp_path), 'gasUGKP', 'unsorted'], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS limited pressure' in result.stdout
