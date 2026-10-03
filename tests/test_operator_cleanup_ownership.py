"""Remaining review items must have one maintained production body."""
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def test_ordered_pressure_uses_shared_face_accumulation():
    source=(ROOT/'common/GpuPressureCellTraversal.cuh').read_text()
    assert 'accumulatePressureFaceDelta(' in source
    assert 'dpx-=sign*s.solidPressurePhiMomX' not in source
def test_cold_wall_input_preparation_has_one_owner():
    source=(ROOT/'common/wall/GpuColdWall1DDevice.cuh').read_text()
    assert source.count('#include "GpuColdWall1DInputs.inl"')==2
    assert 'const GpuReal dPart = clampMin' not in source
@pytest.mark.parametrize('name',['applyCollisionalPressureProjectionCellAtomicKernel','applyCollisionalPressureProjectionParticlesAtomicKernel'])
def test_gas_unused_atomic_kernels_are_removed(name):
    source=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    assert name not in source
