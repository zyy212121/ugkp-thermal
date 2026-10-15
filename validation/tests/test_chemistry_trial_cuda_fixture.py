"""Generate and compile the real chemistry owner fixture; never emulate CUDA.

Generation assertions describe test-support coverage only. Numerical, owner and
WSL host/device equivalence is established by the explicitly enabled GPU run.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = SOURCE_ROOT
PREPARE = (VALIDATION_ROOT / 'tools/tests/fixtures/chemistry_trial/prepare_chemistry_trial.py')


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


def test_fixture_compares_physical_control_statuses_and_independent_inventory(generated_fixture):
    _, source = generated_fixture
    for field in ('code', 'thermoCode', 'reaction', 'attemptedSteps',
                  'rejectedSteps'):
        assert f'sx.{field}' in source and f'sy.{field}' in source
    assert 'left->gasSpecies.faceStatus' in source
    assert 'compareInventory(actual,reference)' in source
    assert 'physicalInventory(const DeviceState* s)' in source
    for field in ('rho[c]*volume', 'rhoE[c]*volume', 'rhoUx[c]*volume',
                  'rhoUy[c]*volume', 'rhoUz[c]*volume', 'rho[k*s->nCells+c]*volume'):
        assert field in source
    for removed in ('gasBudgetAudit', 'gasBudgetHost', 'prepareGasBudgetRequest',
                    'semanticLedger', 'hostReadBytes', 'deviceBytes',
                    'chemistryAudit', 'nonlinearIterations'):
        assert removed not in source
    assert 'maximumSteps=1' in source
    assert 'kind<16' in source
    assert 'kind==15?0.:2e-6' in source
    assert 'g.mechanism.reactants=nullptr;g.chemistryControls.maximumRejectedSteps=100' in source


def test_fixture_covers_complete_failed_transactions_before_public_replay(generated_fixture):
    _, source = generated_fixture
    assert 'void failedTransactionTests()' in source
    assert 'chemistryTrialTest::failedTransactionTests();' in source
    failure = source[source.index('void failedCellTests()'):source.index('void failedTransactionTests()')]
    transaction = source[source.index('void failedTransactionTests()'):source.index('void missingFaceAddressTests()')]
    for member in ('code', 'thermoCode', 'reaction', 'attemptedSteps', 'rejectedSteps'):
        assert f'q.{member}==' in transaction
    assert 'for(int c=0;c<s->nCells;++c)' in transaction
    assert 'bool(successful->gasSpecies.chemistryStatus[0])' in transaction
    assert 'independent successful control step counts changed' in transaction
    assert 'SharedGasTrialPolicy::begin(s)' in transaction
    assert 'SharedGasTrialPolicy::validate(s,true)' in transaction
    assert 'SharedGasTrialPolicy::rollback(s);sync();compare(s,reference);compareInventory(s,reference)' in transaction
    assert 'savedRetry' in transaction and 'savedTerminal' in transaction
    assert 'sharedGasTrialDetail::cellFields(s)' in transaction
    assert 'sharedGasTrialDetail::faceFields(s)' in transaction
    assert 'physicalInventory=1' in transaction
    assert 's->gasSpecies.faceStatus' in transaction
    assert transaction.count('compareInventory(s,reference)') == 2
    assert 'physicalControlStatus=1' in transaction

    controls = '        g.chemistryControls.maximumRejectedSteps=0;'
    last = '        if(kind==15){g.thermo.coefficientCount=1;g.mechanism.stoichiometricBasis=nullptr;}'
    def matrix(text):
        return text[text.index(controls):text.index(last) + len(last)]
    assert matrix(failure) == matrix(transaction)
    assert 'PASS public-replay kind=' in failure
    assert 'savedInventory=physicalInventory(s)' in failure
    assert 'failed native cell changed physical inventory' in failure


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


def test_fixture_real_gpu_owner_state_and_failure_matrix(compiled_fixture):
    if os.environ.get('UGKWP_CHEMISTRY_TRIAL_GPU_TESTS') != '1':
        pytest.skip('Set UGKWP_CHEMISTRY_TRIAL_GPU_TESTS=1 on a real GPU; no CPU fallback')
    result = subprocess.run([str(compiled_fixture / 'chemistry_trial')],
                            capture_output=True, text=True)
    (compiled_fixture / 'run.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_prepare().validate_run_output(result.stdout)['status_rows'] == 32
    assert result.stdout.count('PASS failed-cell kind=') == 16
    assert result.stdout.count('PASS failed-transaction kind=') == 16
    status_rows = re.findall(r'^FAIL_STATUS kind=(\d+) cell=(\d+) ', result.stdout, re.M)
    assert sorted((int(kind), int(cell)) for kind, cell in status_rows) == [
        (kind, cell) for kind in range(16) for cell in range(2)]
    assert set(re.findall(r'PASS public-replay kind=(\d+)', result.stdout)) == {'7', '15'}
    for marker in ('PASS requested owner accepted trace', 'PASS terminal owner completion',
                   'PASS original successful requested interval',
                   'PASS original direct chemistry success'):
        assert marker in result.stdout


def load_prepare():
    import importlib.util
    spec = importlib.util.spec_from_file_location('prepare_chemistry_trial', PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_output():
    lines = [
        'PASS requested owner accepted trace: full interval',
        'PASS terminal owner completion: original public state',
        'PASS original successful requested interval: real owner',
        'PASS original direct chemistry success: native/public fields',
    ]
    for kind in range(16):
        lines.append(f'PASS failed-cell kind={kind} code=1 thermo=0 retry=0 terminal=0')
        lines.append(f'PASS failed-transaction kind={kind} bothCellsFullStatus=1')
        for cell in range(2):
            lines.append(f'FAIL_STATUS kind={kind} cell={cell} code=1 thermo=0 reaction=-1 attempted=0 rejected=0 cellStatus=1')
    for kind in (7, 15):
        lines.append(f'PASS public-replay kind={kind} fullPublicOwner=1 explicitEulerReference=1')
    for label in ('missing-face-owner', 'missing-periodic-pair'):
        for native in (0, 1):
            lines.append(f'PASS {label} native={native} actualMissingResource=1 unchanged=1')
    lines.extend(storage_output().splitlines()[:-1])
    lines.append('PASS real CUDA chemistry Trial routing, owner retry, terminal completion and unavailable-resource controls')
    return '\n'.join(lines)+'\n'


def test_run_validator_requires_complete_control_failure_and_status_matrix():
    summary = load_prepare().validate_run_output(complete_output())
    assert summary == dict(controls=4, failed_cells=16, failed_transactions=16,
                           status_rows=32, public_replays=2, address_controls=4, storage_controls=6)


@pytest.mark.parametrize('missing', [
    'PASS requested owner accepted trace',
    'PASS failed-cell kind=9 ',
    'PASS failed-transaction kind=12 ',
    'FAIL_STATUS kind=3 cell=1 ',
    'PASS public-replay kind=7 ',
    'PASS missing-face-owner native=1 ',
    'PASS missing-periodic-pair native=0 ',
    'PASS storage-failure entry=sst native=1 ',
    'PASS real CUDA chemistry Trial routing',
])
def test_run_validator_rejects_truncated_success_output(missing):
    output = '\n'.join(line for line in complete_output().splitlines()
                       if not line.startswith(missing))
    with pytest.raises(ValueError):
        load_prepare().validate_run_output(output)


def storage_output():
    return '\n'.join(
        f'PASS storage-failure entry={entry} native={native} code=1 unchangedState=1 offsetRestored=1'
        for entry in ('positivity', 'euler', 'sst') for native in (0, 1)
    )+'\nPASS real CUDA storage failure controls\n'


def test_storage_run_validator_requires_each_consumer_and_native_mode():
    assert load_prepare().validate_storage_output(storage_output()) == dict(storage_controls=6)
    with pytest.raises(ValueError):
        load_prepare().validate_storage_output(storage_output().replace(
            'PASS storage-failure entry=sst native=1', 'omitted'))
