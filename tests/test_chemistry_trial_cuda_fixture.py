"""Generate and compile the real chemistry owner fixture; never emulate CUDA.

Generation assertions describe test-support coverage only. Numerical, owner and
WSL host/device equivalence is established by the explicitly enabled GPU run.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / 'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py'


@pytest.fixture(scope='module')
def generated_fixture(tmp_path_factory):
    output = tmp_path_factory.mktemp('chemistry-trial-generated')
    result = subprocess.run([sys.executable, str(PREPARE), str(ROOT), str(output),
                             '--generate-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output, (output / 'chemistry_trial.cu').read_text()


def test_fixture_separates_canonical_host_owner_and_device_view(generated_fixture):
    _, source = generated_fixture
    assert 'new DeviceState' in source
    assert 'cudaMalloc(&s->deviceState,sizeof(DeviceState))' in source
    assert 's->deviceState=s' not in source
    assert 'cudaMemcpy(s->deviceState,s,sizeof(DeviceState),cudaMemcpyHostToDevice)' in source
    assert 'advanceGasChemistryKernel<<<1,32>>>(s->deviceState' in source
    assert 'advanceGasChemistryKernel<<<1,32>>>(s,' not in source
    assert 'ALLOCATE_NATIVE_FIELDS' not in source


def test_fixture_uses_real_courant_owner_and_committed_duration_replay(generated_fixture):
    _, source = generated_fixture
    assert 'ugkwpGpuResidentStrictComputeGasCourant(s,dt,.5,time,&co)' in source
    assert 'TracePolicy::acceptedDurations' in source
    assert 'advanceGasTrial<SharedGasTrialPolicy>(reference,dt,time+elapsed,true)' in source
    assert 'elapsed==requestedDt' in source
    assert 'original successful requested interval' in source
    assert 'original direct chemistry success' in source
    assert 'TracePolicy::sourceDurations[0]==2e-6' not in source


def test_fixture_compares_all_semantic_audits_statuses_and_budgets(generated_fixture):
    _, source = generated_fixture
    for field in ('speciesMassChange', 'massResidual', 'maximumElementResidual',
                  'energyResidual', 'maximumLocalError', 'integratedTime',
                  'acceptedSteps', 'rejectedSteps', 'nonlinearIterations'):
        assert f'x.{field}' in source and f'y.{field}' in source
    for field in ('code', 'thermoCode', 'reaction', 'attemptedSteps',
                  'rejectedSteps', 'nonlinearIterations'):
        assert f'sx.{field}' in source and f'sy.{field}' in source
    assert 'left->gasSpecies.faceStatus' in source
    assert 'compareBudget(actual,reference)' in source
    assert 'semanticLedger(left)==semanticLedger(right)' in source
    assert 'maximumSteps=1' in source
    assert 'kind<16' in source
    assert 'kind==15?0.:2e-6' in source
    assert 'g.mechanism.reactants=nullptr;g.chemistryControls.maximumRejectedSteps=100' in source


@pytest.fixture(scope='module')
def compiled_fixture(tmp_path_factory):
    nvcc = os.environ.get('NVCC') or shutil.which('nvcc')
    if not nvcc:
        pytest.skip('nvcc unavailable; no native CUDA compile/runtime claim')
    output = tmp_path_factory.mktemp('chemistry-trial-native')
    result = subprocess.run([sys.executable, str(PREPARE), str(ROOT), str(output),
                             '--nvcc', nvcc], capture_output=True, text=True)
    build_log = (output / 'build.log').read_text() if (output / 'build.log').exists() else ''
    assert result.returncode == 0, result.stdout + result.stderr + build_log
    return output


def test_fixture_compiles_complete_production_cuda_owner(compiled_fixture):
    assert (compiled_fixture / 'chemistry_trial').is_file()
    command = json.loads((compiled_fixture / 'command.json').read_text())
    assert '-O3' in command and '--fmad=true' in command


def test_fixture_real_gpu_owner_audits_and_failure_matrix(compiled_fixture):
    if os.environ.get('UGKWP_CHEMISTRY_TRIAL_GPU_TESTS') != '1':
        pytest.skip('Set UGKWP_CHEMISTRY_TRIAL_GPU_TESTS=1 on a real GPU; no CPU fallback')
    result = subprocess.run([str(compiled_fixture / 'chemistry_trial')],
                            capture_output=True, text=True)
    (compiled_fixture / 'run.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('PASS failed-cell kind=') == 16
    for marker in ('PASS requested owner accepted trace', 'PASS terminal owner completion',
                   'PASS original successful requested interval',
                   'PASS original direct chemistry success'):
        assert marker in result.stdout
