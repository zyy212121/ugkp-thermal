from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def function_body(source: str, signature: str, next_signature: str) -> str:
    return source.split(signature, 1)[1].split(next_signature, 1)[0]


def test_radiation_updates_mobile_particles_only() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    validation = function_body(
        cuda,
        "__global__ void validateParticleRadiationAffineTemperatureKernel",
        "__global__ void applyParticleRadiationAffineTemperatureKernel",
    )
    application = function_body(
        cuda,
        "__global__ void applyParticleRadiationAffineTemperatureKernel",
        "__device__ void clearSolidCell",
    )
    for body in (validation, application):
        assert "s.pStuck[particleI] != Foam::gpuThermal::particleWallMobile" in body


def test_mobile_radiation_snapshot_uses_packed_base_or_enthalpy_free_atomic_path() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    body = function_body(
        cuda,
        'extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums',
        'extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea',
    )
    assert "preBaseDirectoryReady" in body
    assert "accumulatePackedMobileParticleRadiationSumsKernel" in body
    assert "accumulateMobileParticleRadiationSumsAtomicKernel" in body
    assert "binParticlesByCell" not in body
    assert "rebuildResidentParticleMomentsFromParticles" not in body


def test_post_radiation_refresh_is_enthalpy_only_and_preserves_eps_history() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    body = function_body(
        cuda,
        'extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked',
        'extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev',
    )
    assert "refreshPackedParticleEnthalpyKernel" in body
    assert "accumulateParticleEnthalpyMomentAtomicKernel" in body
    assert "recoverParticleEnthalpyMomentAtomicKernel" in body
    for forbidden in (
        "binParticlesByCell",
        "clearParticleMomentsKernel",
        "solidRecoveryFromParticleMomentsKernel",
        "initialiseEpsGPrevKernel",
        "initialiseThetaDragAlphaKernel",
        "rebuildResidentParticleMomentsFromParticles",
    ):
        assert forbidden not in body


def test_wall_radiation_uses_instantaneous_exposed_area() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    coupler = read("thermal/GpuSolidThermalCoupler.C")
    occupied = function_body(
        cuda,
        'extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea',
        'extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked',
    )
    assert "prepareParticleWallContactAreaScale" in occupied
    assert "represented*scale" in occupied
    assert "particleWallOccupiedContactArea" in coupler
    assert "emissivity[patchI][faceI] *= exposedFraction" in coupler


def test_legacy_full_rebuild_keeps_restart_history_initialisation_separate() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    full_rebuild = function_body(
        cuda,
        "int rebuildResidentParticleMomentsFromParticles",
        "__global__ void solidRecoveryFromParticleMomentsKernel",
    )
    assert "initialiseEpsGPrevKernel" in full_rebuild
    assert "ugkwpGpuResidentStrictRebuildParticleMomentsPacked" not in cuda
