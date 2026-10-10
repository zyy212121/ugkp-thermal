"""Coupling-only frontend and shared numerical ownership regression gates."""
from pathlib import Path
import subprocess

APP = Path(__file__).resolve().parents[1]

def test_private_gas_solver_is_not_restored():
    for name in ('gas/AleFlux.H', 'gas/GasKernels.cu', 'gas/SstKernels.cu',
                 'gpu/GasWindowImplementation.cuh'):
        assert not (APP / name).exists(), name
    for path in APP.rglob('*'):
        if path.suffix in ('.H', '.C', '.cu', '.cuh'):
            text = path.read_text()
            assert 'aleRusanov(' not in text, path
            assert '#include "gas/AleFlux.H"' not in text, path

def test_coupling_entry_rejects_pure_flow(tmp_path):
    binary = tmp_path / 'entry'
    subprocess.run(['g++', '-std=c++14', '-Wall', '-Wextra', '-Werror', '-pedantic',
                    '-I' + str(APP), str(APP / 'tests/test_coupling_entry.cpp'),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

def test_thin_stage_adapter_uses_actual_shared_host_owner():
    text = (APP / 'gpu/SharedGasAdvance.cuh').read_text()
    assert '#include "../../../common/GpuGasAdvance.cuh"' in text
    assert 'advanceGasFluxStage(' in text
    assert '<<<' not in text
    assert 'midpoint' not in text.lower()
