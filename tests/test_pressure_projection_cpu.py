"""Host execution of production pressure writers and restored local closure.

These replace transaction/preflight assertions that are intentionally no longer
part of the pressure contract. CUDA scheduling is covered separately.
"""
from pathlib import Path
import json
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODES = ['particle_clamp', 'closure_clamp', 'cold_initial', 'nonfinite_particle',
         'sorted', 'compact', 'split', 'segments', 'unsorted',
         'zero_sorted', 'zero_compact', 'zero_split', 'zero_unsorted',
         'mixed_sorted', 'mixed_compact', 'mixed_split', 'mixed_unsorted',
         'limited_faces', 'invalid_face', 'nonfinite_delta',
         'scaling_subnormals', 'scaling_boundaries', 'scaling_boundaries_ftz']

@pytest.fixture(scope='module', params=['float', 'double'])
def executable(request, tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ is required')
    binary = tmp_path_factory.mktemp('pressure_projection_' + request.param) / 'check'
    run = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
        '-Wno-unused-parameter', '-DTEST_REAL=' + request.param,
        '-I', str(ROOT / 'common'), str(ROOT / 'tests/fixtures/pressure/projection_cpu.cpp'),
        '-o', str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return binary

@pytest.mark.parametrize('mode', MODES)
def test_production_pressure_projection(executable, mode):
    run = subprocess.run([str(executable), mode], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_pressure_pipeline_has_no_preview_transaction():
    pipeline = (ROOT / 'common/GpuPressurePipeline.cuh').read_text()
    for removed in ('launchPressurePreflight', 'publishPressureCanonicalMomentsKernel',
                    'pressureFailure', 'cudaMemsetAsync', 'cudaDeviceSynchronize'):
        assert removed not in pipeline
    for app, folder in [('gasUGKP', 'private_backend'), ('FSH', 'private_backend'), ('CHT', 'gpu')]:
        source = (ROOT / 'applications' / app / folder / 'GpuResidentStrict.cu').read_text()
        for removed in ('pressurePreviewMoments', 'pressureFailure', 'GpuPressurePreflight.cuh',
                        'readValidatedPressureDelta', 'pressureDeltaIsZero'):
            assert removed not in source, (app, removed)
    assert not (ROOT / 'common/GpuPressurePreflight.cuh').exists()
    assert not (ROOT / 'common/GpuPressureFailure.cuh').exists()
    mirrors = json.loads((ROOT / 'tools/managed_mirrors.json').read_text())['mirrors']
    declared = {entry['path'] for entry in mirrors}
    assert 'common/GpuPressurePreflight.cuh' not in declared
    assert 'common/GpuPressureFailure.cuh' not in declared
