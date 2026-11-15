"""Keep removed reporting-only work out of actual CHT exchange paths."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path


THERMAL = (SOURCE_ROOT / 'applications/CHT/thermal')


def test_radiation_publishes_raw_boundary_exchange_without_balancing():
    source = (SOURCE_ROOT / "common/gpu/ParticleRadiationMaterial.cu").read_text()
    publication = source.split("__global__ void publishMaterialBoundary", 1)[1].split("__global__ void settleMaterialBoundary", 1)[0]
    settlement = source.split("__global__ void settleMaterialBoundary", 1)[1].split("struct ParticleRadiationMaterial::Implementation", 1)[0]
    assert "coupled[face]?raw[face]:0" in publication
    assert "other+=raw[face]" in publication
    assert "*acceptedTotal=particleHeat" in settlement
    assert "boundary[" not in settlement and "*escaped" not in settlement
    assert "conservativeBoundaryEnergyScale" not in source


def test_contact_energy_has_no_reporting_only_accumulator_or_restart_slot():
    files = ["GpuSolidThermalCoupler.C", "GpuThermalExchangeState.C",
             "GpuThermalExchangeState.H"]
    source = "\n".join((THERMAL / name).read_text() for name in files)
    assert "particleContactEnergyJ" not in source
    backend = (SOURCE_ROOT / "applications/CHT/gpu/GpuWallEnergy64.cu").read_text()
    assert "combined+=deposited+reflected" in backend
    assert "previewInterfaceExchange" in source
