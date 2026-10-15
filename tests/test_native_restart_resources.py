from pathlib import Path
import importlib.util
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tests/fixtures/native_restart_resources/prepare_native_restart_resources.py'


def load_runner():
    spec = importlib.util.spec_from_file_location('native_restart_resources', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('app', ['CHT', 'FSH', 'gasUGKP'])
def test_generated_driver_matches_real_native_api_types(tmp_path, app):
    runner = load_runner()
    source, fields = runner.generate(ROOT, app)
    assert fields
    path = tmp_path / 'native_restart.cpp'
    path.write_text(source)
    result = subprocess.run(['g++', '-std=c++17', '-I/usr/local/cuda/include',
                             '-c', str(path), '-o', str(tmp_path / 'native_restart.o')],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_run_validator_requires_complete_actual_api_matrix():
    runner = load_runner()
    fields = [('double', 'px'), ('int', 'pStatus')]
    rows = ['PASS native-restart case=' + case + ' unchanged=1' for case in
            ['negative-count', 'over-capacity', 'null-px', 'null-pStatus']]
    output = '\n'.join(rows + ['PASS native-restart app=gasUGKP fields=2 retry=1 coldOptionalNull=1 actualCuda=1'])
    assert runner.validate_run_output(output, 'gasUGKP', fields)['invalid_inputs'] == 4
    with pytest.raises(ValueError):
        runner.validate_run_output(output.replace(rows[0], ''), 'gasUGKP', fields)
