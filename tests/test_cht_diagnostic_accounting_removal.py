"""Keep removed reporting-only work out of actual CHT exchange paths."""
from pathlib import Path


THERMAL = Path(__file__).resolve().parents[1] / "applications/CHT/thermal"


def test_radiation_keeps_only_the_physical_boundary_scale():
    source = (THERMAL / "GpuParticleRadiationCoupler.C").read_text()
    header = (THERMAL / "GpuParticleRadiationCoupler.H").read_text()
    assert source.count("detail::conservativeBoundaryEnergyScale") == 1
    assert "boundaryScale.scale*rawBoundaryEnergyJ[patchI][faceI]" in source
    assert "appliedBoundaryEnergyJ" not in source
    assert "globalConservationResidualJ" not in source + header


def test_contact_energy_has_no_reporting_only_accumulator_or_restart_slot():
    files = ["GpuSolidThermalCoupler.C", "GpuThermalExchangeState.C",
             "GpuThermalExchangeState.H"]
    source = "\n".join((THERMAL / name).read_text() for name in files)
    assert "particleContactEnergyJ" not in source
    assert "particleContactPreview += particleReflectedPreview;" in source
    assert "combined[faceI] += particleContactPreview" in source
