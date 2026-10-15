from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('branch', ['gasUGKP', 'FSH', 'CHT'])
def test_split_pressure_uses_limited_face_flux(tmp_path, branch):
    """Local FV balance and particle/Eulerian closure, with base plus injection."""
    if not (ROOT / "applications" / branch).is_dir():
        pytest.skip("Application is not included in this package")
    if not Path("/usr/local/cuda/bin/nvcc").is_file():
        pytest.skip("CUDA compiler required")
    result = subprocess.run(
        ["python3", str(ROOT / "tests/fixtures/pressure/limited_flux_balance.py"),
         str(ROOT), str(tmp_path), branch],
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS limited pressure" in result.stdout
