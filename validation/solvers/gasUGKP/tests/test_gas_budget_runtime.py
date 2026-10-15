"""Deleted production audit hooks must add no resources, I/O, or admission.

Numerical transaction and rollback coverage remains in the native trial tests.
These probes execute the actual adapter with a host-memory CUDA shim.
"""
from pathlib import Path
import os
import subprocess
import pytest
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT


def test_production_budget_runtime_is_removed():
    private = ROOT / 'applications/gasUGKP/private_backend'
    assert not (private / 'GasBudgetRuntime.cuh').exists()
    assert not (ROOT / 'common/gasTransport/GasBudgetAudit.H').exists()
    for name in ('GpuResidentStrict.cu', 'SharedGasAdapter.cuh', 'SharedGasTrialPolicy.cuh'):
        source = (private / name).read_text()
        for removed in ('gasBudgetHost', 'gasBudgetAudit', 'prepareGasBudgetRequest',
                        'beginGasBudgetInterval', 'commitGasBudgetInterval', 'cancelGasBudgetInterval'):
            assert removed not in source, (name, removed)


@pytest.mark.parametrize('destination', ['missing_parent', 'existing_file', 'directory'])
def test_removed_output_setting_cannot_change_adapter_execution(tmp_path, destination):
    source = tmp_path / 'adapter.cpp'
    source.write_text(ADAPTER_PROBE)
    binary = tmp_path / 'adapter'
    build = subprocess.run(['g++', '-std=c++17', '-O0', '-I'+str(ROOT), str(source), '-o', str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    path = tmp_path / destination
    if destination == 'missing_parent':
        path = path / 'ledger.jsonl'
    elif destination == 'existing_file':
        path.write_text('preserve this file\n')
    else:
        path.mkdir()
    clean_env = {k:v for k,v in os.environ.items() if k != 'UGKP_GAS_BUDGET_OUTPUT'}
    normal = subprocess.run([str(binary)], env=clean_env, capture_output=True, text=True)
    configured = subprocess.run([str(binary)], env={**clean_env, 'UGKP_GAS_BUDGET_OUTPUT': str(path)}, capture_output=True, text=True)
    assert normal.returncode == configured.returncode == 0, normal.stderr + configured.stderr
    assert (normal.stdout, normal.stderr) == (configured.stdout, configured.stderr)
    if destination == 'missing_parent':
        assert not path.exists()
    elif destination == 'existing_file':
        assert path.read_text() == 'preserve this file\n'
    else:
        assert list(path.iterdir()) == []
