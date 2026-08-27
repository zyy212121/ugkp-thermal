from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ugkpcht_has_only_local_fourier_wall_heat_transfer():
    source = (ROOT / "gpu" / "GpuResidentStrict.cu").read_text()
    header = (ROOT / "gpu" / "GpuResidentStrict.H").read_text()
    frontend = (ROOT / "diluteUgkwpFoam.C").read_text()

    combined = source + header + frontend
    for forbidden in (
        "Bartz",
        "bartz",
        "gasWallHeatTransferModel",
        "configureGasWallHeatTransfer",
        "ConfigureGasWallHeatTransfer",
    ):
        assert forbidden not in combined

    assert "energyFlux -= kEffective*normalTemperatureGradient;" in source
    assert "const double thermalAlpha = kEffective/(rhoSafe*s.gasCp + OfSmall);" in source
    assert "fmax(nu, thermalAlpha)*s.magSf[f]*s.deltaCoeffs[f]" in source


def test_mss7_case_does_not_select_a_wall_heat_flux_correlation():
    properties = (
        ROOT.parents[1]
        / "examples/thermal/MSS7_laminar/constant/particleProperties"
    ).read_text()
    assert "gpuResidentGasWallHeatTransfer" not in properties
    assert "Bartz" not in properties
