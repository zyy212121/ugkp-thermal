from pathlib import Path
import subprocess
import pytest
ROOT = Path(__file__).resolve().parents[1]

def test_unsorted_pressure_preserves_limited_flux_and_moments(tmp_path):
    """Exercise the reachable unsorted CUDA pipeline, including its shared dependencies."""
    if not Path('/usr/local/cuda/bin/nvcc').is_file():
        pytest.skip('CUDA compiler required')
    result = subprocess.run(['python3', str(ROOT/'tests/fixtures/pressure/limited_flux_balance.py'),
        str(ROOT), str(tmp_path), 'gasUGKP', '64', 'unsorted'], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS limited pressure' in result.stdout
