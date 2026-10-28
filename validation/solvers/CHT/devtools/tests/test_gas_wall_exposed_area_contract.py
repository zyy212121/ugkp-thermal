from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path


SOURCE = (SOURCE_ROOT / 'common/thermal/GpuResidentStrict.cu')


def source_text():
    return SOURCE.read_text(encoding="utf-8")


def function_body(text, signature, next_signature):
    start = text.index(signature)
    end = text.index(next_signature, start)
    return text[start:end]


def test_contact_area_is_converted_to_exposed_fraction_on_device():
    body=(SOURCE_ROOT/'common/GpuGasWallExposedArea.cuh').read_text()
    formula=(SOURCE_ROOT/'common/gasTransport/WallHeatExposure.H').read_text()
    assert "s.particleStuckCandidateMask[f] == 0" in body
    assert "s.particleWallRepresentedContactArea[f]" in body
    assert "s.particleWallContactAreaScale[f]" in body
    assert "ugkwp::wallHeatExposedFraction" in body
    assert "Real(1)-occupied/faceArea" in formula


def test_only_wall_thermal_energy_flux_is_area_scaled():
    body=(SOURCE_ROOT/'common/operators/computeRiemannGasFaceFluxDevice.cuh').read_text()
    assert "wallThermalAreaFraction*directWallHeatFlux" in body
    assert "wallThermalAreaFraction\n                   *kEffective*normalTemperatureGradient" in body
    assert "momentumFluxX -= traction.x;" in body
    assert "wallThermalAreaFraction*traction" not in body


def test_contact_area_is_prepared_before_gas_flux_without_duplicate_particle_call():
    text = source_text()
    body = function_body(
        text,
        'extern "C" int ugkwpGpuResidentStrictAdvance',
        'extern "C" int ugkwpGpuResidentStrictDownload',
    )
    preparation = body.index("prepareParticleWallContactAreaScale(s, block)")
    gas_advance = body.index("advanceGasFluxStage(s, dt, simulationTime-dt)")
    collision = body.index("UGKP_DEV_PROBE_ENTER(ProbeCollisionPool)")
    assert preparation < gas_advance < collision
    assert body.count("prepareParticleWallContactAreaScale(s, block)") == 1
