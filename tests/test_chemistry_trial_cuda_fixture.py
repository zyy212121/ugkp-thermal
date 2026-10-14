"""Generate and compile the real chemistry owner fixture; never emulate CUDA.

Generation assertions describe test-support coverage only. Numerical, owner and
WSL host/device equivalence is established by the explicitly enabled GPU run.
"""
import json
import os
import re
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
    transaction = source[source.index('void failedTransactionTests()'):source.index('void cleanup()')]
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
    # Both loops must keep the original 16 model/backing/max-step combinations.
    controls = '        g.chemistryControls.maximumRejectedSteps=0;'
    last = '        if(kind==15){g.thermo.coefficientCount=1;g.mechanism.stoichiometricBasis=nullptr;}'
    def matrix(text):
        return text[text.index(controls):text.index(last) + len(last)]
    assert matrix(failure) == matrix(transaction)
    # In particular, log the native failure before kinds 7/15 reset its status.
    assert failure.index('PASS failed-cell kind=') < failure.index('reset metadata control')
    assert 'PASS public-replay kind=' in failure
    assert 'comparePhysicalFields(s,replayReference);compareInventory(s,replayReference)' in failure
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
