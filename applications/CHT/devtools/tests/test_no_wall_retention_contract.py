from __future__ import annotations

from pathlib import Path
import pytest


pytestmark = pytest.mark.skip(
    reason=(
        "historical no-retention CHT contract is superseded by the requested "
        "shared FSH/CHT finite-contact and deposition state machine"
    )
)


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_particle_state_has_no_wall_retention_metadata() -> None:
    source = read("gpu/GpuResidentStrict.cu")
    header = read("gpu/GpuResidentStrict.H")
    thermal = read("thermal/GpuParticleWallContactHeat.H")
    combined = source + header + thermal

    for forbidden in (
        "pStuckFaceId",
        "pAdep",
        "compactPStuckFaceId",
        "compactPAdep",
        "particleWallDepositionMask",
        "particleWallRepresentedContactArea",
        "particleWallContactAreaScale",
        "depositedParticleWallContact",
        "representedDepositionContactArea",
        "depositionContactAreaScale",
        "particleWallMaximumDepositionCoverage",
    ):
        assert forbidden not in combined


def test_all_physical_walls_use_the_reflection_path() -> None:
    source = read("gpu/GpuResidentStrict.cu")
    assert "else if (kind == 1 || kind == 2)" in source
    assert "s.cellFaceRestitution[plane]" in source


def test_reflected_impact_heat_transfer_is_retained() -> None:
    source = read("gpu/GpuResidentStrict.cu")
    thermal = read("thermal/GpuParticleWallContactHeat.H")
    assert "applyParticleWallImpactHeat" in source
    assert "impactParticleWallContact" in source
    assert "atomicAdd(&s.particleWallEnergy[globalFaceId], result.wallEnergyJ);" in source
    assert "integratedContactAreaM2S" in thermal
    assert "hcImpact" in thermal


def test_restart_schema_contains_mobile_particle_state_only() -> None:
    header = read("gpu/GpuResidentStrict.H")
    thermal_state = read("thermal/GpuThermalExchangeState.C")
    assert "UGKP_PARTICLES_SCHEMA4" in header
    assert "UGKP_PARTICLES_SCHEMA4" in thermal_state
    assert "UGKP_PARTICLES_SCHEMA1_BIN" in header
    assert "UGKP_PARTICLES_SCHEMA1_BIN" in thermal_state
    assert "UGKP_PARTICLES_SCHEMA3" not in header + thermal_state
