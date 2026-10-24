from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from source_contract_utils import read_source
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')


def read(relative: str) -> str:
    return read_source(ROOT / relative)


def test_frontend_builds_pure_translation_pair_topology():
    source = read("gpu/GpuResidentStrict.H")
    assert '#include "cyclicPolyPatch.H"' in source
    assert "facePeriodicPair" in source
    assert "facePeriodicDx" in source
    assert "transformGlobalFace" in source
    assert "transform().transforms()" in source
    assert "serial pure-translation cyclic" in source
    assert "kind = 5" in source


def test_mesh_topology_crosses_the_typed_resident_api():
    for relative in ('gpu/GpuBackendApi.H','gpu/GpuResidentStrict.H','private_backend/GpuResidentStrict.cu'):
        text=(ROOT/relative).read_text()
        for field in ('facePeriodicPair','facePeriodicDx','facePeriodicDy','facePeriodicDz'):
            assert field in text



def test_backend_maps_periodic_gas_and_particle_transport():
    source = read("private_backend/GpuResidentStrict.cu")
    # The gradient helper moved out of the resident; inspect the included owner.
    assert '#include "operators/riemannFacePrimitiveForGradient.cuh"' in source
    source += "\n" + read("../../common/operators/riemannFacePrimitiveForGradient.cuh")
    assert "isPeriodicFace" in source
    assert "periodicMappedCellCentre" in source
    assert "enforcePeriodicGasFluxAntisymmetryKernel" in source
    assert "kind == 5" in source
    assert "s.facePeriodicDx[faceI]" in source
    assert "s.facePeriodicPair[f]" in source


def test_split_dpre_lifecycle_is_still_explicit():
    source = read("private_backend/GpuResidentStrict.cu")
    assert "preBaseDirectoryReady" in source
    assert "prepareCsrHeavyBaseReductionTasks" in source
    assert "if (binParticlesByCell(s, block) != 0)" in source


def test_scope_remains_fail_closed():
    source = read("gpu/GpuResidentStrict.H")
    assert "rotational" in source
    assert "cpp.transform().transforms()" in source
    for consumed_address in ("pairFaceI < nInternalFaces", "pairFaceI >= nFaces", "owner[pairFaceI] < 0", "owner[pairFaceI] >= nCells"):
        assert consumed_address in source
