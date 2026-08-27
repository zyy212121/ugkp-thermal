from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "thermal" / "GpuSolidThermalCoupler.C"


def test_coupled_wall_flux_fields_are_registered_and_written():
    text = SOURCE.read_text()
    assert '"gasConvectiveWallHeatFlux"' in text
    assert '"particleRadiationWallHeatFlux"' in text
    assert "particleStuckWallHeatFluxFieldName()" in text
    assert "particleReflectedWallHeatFluxFieldName()" in text
    assert text.count("IOobject::AUTO_WRITE") >= 4


def test_coupled_wall_flux_fields_restore_the_last_committed_restart_values():
    text = SOURCE.read_text()
    field_block = text[text.index("gasConvectiveWallHeatFlux.reset"):text.index("if (resident.particleWallHeatTransferEnabled())")]
    assert field_block.count("IOobject::READ_IF_PRESENT") == 4
    assert "IOobject::NO_READ" not in field_block


def test_fluxes_use_their_own_integrated_ledger_intervals():
    text = SOURCE.read_text()
    assert "energy[faceI]/(area[faceI]*deltaTExchange)" in text
    assert "energy[faceI]/(area[faceI]*elapsedSinceRadiation)" in text
    assert "radiation().solidWallRadiationEnergyJ[patchI]" in text
    assert "peekParticleWallHeatLedgers" in text
    assert "downloadAndResetParticleWallHeatLedgers" in text
    assert "particleDepositedPreview[globalFaceI]" in text
    assert "particleReflectedPreview[globalFaceI]" in text
