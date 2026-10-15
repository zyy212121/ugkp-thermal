import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get('UGKWP_WALL_SLOT_GPU_TESTS') != '1',
                    reason='Set UGKWP_WALL_SLOT_GPU_TESTS=1 for actual CUDA and memcheck')
def test_actual_wall_slot_resource_bounds_and_retry(tmp_path):
    nvcc = os.environ.get('NVCC') or shutil.which('nvcc')
    assert nvcc, 'Actual CUDA compiler required; no host fallback'
    prepare = ROOT / 'tests/fixtures/wall_slot_resources/prepare_wall_slot_resources.py'
    command = [sys.executable, str(prepare), str(ROOT), str(tmp_path),
               '--nvcc', nvcc, '--memcheck']
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    logs = '\n'.join(path.read_text() for path in sorted(tmp_path.glob('*.log')))
    assert result.returncode == 0, result.stdout + result.stderr + logs
