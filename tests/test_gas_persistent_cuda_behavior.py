"""Exercise the actual pool kernels, queue resets and selection state on CUDA."""
from pathlib import Path
import subprocess, shutil
import pytest
ROOT = Path(__file__).resolve().parents[1]

def test_gas_nonempty_queue_reuse_and_exact_coverage(tmp_path):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():
        pytest.skip('CUDA compiler required')
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    if not Path(smi).is_file() or subprocess.run([smi, '-L'], capture_output=True).returncode:
        pytest.skip('CUDA GPU required')
    q = subprocess.run(['python3', str(ROOT / 'tests/fixtures/shared_operators/collision_behavior.py'),
                        str(ROOT), str(tmp_path), 'gasUGKP', '64'], capture_output=True, text=True, timeout=240)
    assert q.returncode == 0, q.stdout + q.stderr
    assert 'PASS collision pool CPU moments' in q.stdout
