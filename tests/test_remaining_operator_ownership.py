"""Architectural contracts: numerical bodies have one maintenance location.

These checks complement executable conservation tests; they cannot prove
numerical equivalence. Local launch/precision/physics adapters are allowed.
"""
from pathlib import Path
import re
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app,name', [
 ('CHT','solidEpsFromMomentDevice'),('CHT','solidPressureFromMomentsDevice'),
 ('CHT','granularCollisionTauFromCellDevice'),
 ('gasUGKP','riemannFacePrimitiveForGradient'),
 ('FSH','atomicAddParticleWallEnergyByFace'),('CHT','atomicAddParticleWallEnergyByFace')])
def test_numerical_body_is_owned_by_common(app,name):
    leaf='gpu' if app=='CHT' else 'private_backend'
    source=(ROOT/'applications'/app/leaf/'GpuResidentStrict.cu').read_text()
    assert not re.search(r'__device__[^;{}]*?\b'+name+r'\s*\(',source), 'duplicated device computation: '+name
    assert (ROOT/'common/operators'/(name+'.cuh')).is_file()
def test_unsorted_face_and_reconstruction_algebra_has_one_owner():
    shared=(ROOT/'common/GpuPressureUnsortedAlgebra.cuh')
    assert shared.is_file(), 'missing common unsorted pressure algebra'
    assert 'recoverUnsortedPressureKinematics' in shared.read_text()
def test_cold_wall_advance_body_has_one_owner():
    source=(ROOT/'common/wall/GpuColdWall1DDevice.cuh').read_text()
    assert 'candidateEnthalpy = oldEnthalpy' not in source, 'duplicated precision branch computation'
    assert source.count('#include "GpuColdWall1DAdvance.inl"')==2
def test_drag_rate_body_has_explicit_compile_time_policies():
    source=(ROOT/'common/gasNumerics/GpuDragAlgebra.cuh').read_text()
    assert 'struct RegularizedDragInputs' in source
    assert 'struct BoundedDragInputs' in source
    assert 'inverseSchillerNaumannTime<' in source
    assert 'inverseGidaspowTime<' in source
