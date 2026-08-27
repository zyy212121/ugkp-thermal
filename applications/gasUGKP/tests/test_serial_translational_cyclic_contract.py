from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_frontend_builds_pure_translation_pair_topology():
    source = read("gpu/GpuResidentStrict.H")
    assert '#include "cyclicPolyPatch.H"' in source
    assert "facePeriodicPair" in source
    assert "facePeriodicDx" in source
    assert "transformGlobalFace" in source
    assert "transform().transforms()" in source
    assert "serial pure-translation cyclic" in source
    assert "kind = 5" in source


def test_mesh_payload_crosses_the_separated_boundary():
    api = read("gpu/GpuBackendApi.H")
    client = read("gpu/GpuBackendClient.C")
    server = read("private_backend/GpuBackendServer.C")
    protocol = read("gpu/GpuBackendProtocol.H")
    for text in (api, client, server):
        assert "facePeriodicPair" in text
        assert "facePeriodicDx" in text
        assert "facePeriodicDy" in text
        assert "facePeriodicDz" in text
    assert "protocolMinor = 0" in protocol


def test_backend_maps_periodic_gas_and_particle_transport():
    source = read("private_backend/GpuResidentStrict.cu")
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
    for unsupported in ("rotational", "one-to-one conformal"):
        assert unsupported in source
    assert "cpp.transform().transforms()" in source
    assert "cpp.size() != cpp.nbrPatch().size()" in source
