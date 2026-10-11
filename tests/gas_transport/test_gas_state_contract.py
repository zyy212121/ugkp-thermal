"""Check that the gas-only state manifest covers the actual common consumers."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
GAS_HEADERS = (
    "makeGasPrimDevice", "riemannFacePrimitiveForGradient",
    "computeGasPrimitiveGradientsKernel", "computeSstGradientsKernel",
    "sstVelocityInvariants", "computeGasHllcAdcSensorKernel",
    "computeGasGradientLimiterKernel", "computeGasEddyViscosityKernel",
    "updateWaveTransmissivePressureBoundaryKernel",
    "updateLegacyGasBoundaryMirrorKernel", "gasFaceSubgridTransportProperties",
    "computeRiemannGasFaceFluxDevice", "computeGasInternalFaceFluxKernel",
    "computeGasFluxPositivityScaleKernel", "computeSstFaceFluxKernel",
    "linearScheduledValueDevice",
)

def test_manifest_covers_actual_state_dependencies_and_excludes_particle_resources():
    actual = set()
    for header in GAS_HEADERS:
        source = (ROOT / "common/operators" / (header + ".cuh")).read_text()
        actual.update(re.findall(r"\bs\.(\w+)", source))
        assert not re.search(r"\bDeviceState\b", source), header
    manifest = (ROOT / "common/gasTransport/GasStateFields.inc").read_text()
    fields = set(re.findall(r"UGKWP_GAS_STATE_FIELD\([^,]+,\s*(\w+)\)", manifest))
    assert fields | {"gasSpecies", "gasThermalConductivity", "gasGeometry", "gasSstAudit"} == actual
    assert not any(re.search(r"particle|pool|packing|csr|cuda|Stream|Graph", f, re.I) for f in fields)

def test_all_three_legacy_consumers_reach_the_same_operator_owners():
    for app, leaf in (("gasUGKP", "private_backend"), ("FSH", "private_backend"), ("CHT", "gpu")):
        if app != "gasUGKP" and not (ROOT / "applications" / app).is_dir():
            continue
        source = (ROOT / "applications" / app / leaf / "GpuResidentStrict.cu").read_text()
        assert '#include "../../../common/GpuGasAdvance.cuh"' in source
        for header in GAS_HEADERS:
            assert f'#include "operators/{header}.cuh"' in source, (app, header)
            assert not re.search(r"__(?:global|device)__[^;{}]*\b" + header + r"\s*\(", source), (app, header)
