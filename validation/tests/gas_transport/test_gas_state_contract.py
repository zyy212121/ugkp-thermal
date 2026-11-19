"""Check that the gas-only state manifest covers the actual common consumers."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import re

ROOT = SOURCE_ROOT
GAS_HEADERS = (
    "makeGasPrimDevice", "riemannFacePrimitiveForGradient",
    "computeGasPrimitiveGradientsKernel", "computeSstGradientsKernel",
    "sstVelocityInvariants", "computeGasHllcAdcSensorKernel",
    "computeGasGradientLimiterKernel", "computeGasEddyViscosityKernel",
    "updateWaveTransmissivePressureBoundaryKernel",
    "updateLegacyGasBoundaryMirrorKernel", "gasFaceSubgridTransportProperties",
    "computeRiemannGasFaceFluxDevice", "computeGasInternalFaceFluxKernel",
    "computeGasFluxPositivityScaleKernel", "computeSstFaceFluxKernel",
)

def test_manifest_covers_actual_state_dependencies_and_excludes_particle_resources():
    actual = set()
    for header in GAS_HEADERS:
        source = (ROOT / "common/operators" / (header + ".cuh")).read_text()
        actual.update(re.findall(r"\bs\.(\w+)", source))
        assert not re.search(r"\bDeviceState\b", source), header
    manifest = (ROOT / "common/gasTransport/GasStateFields.inc").read_text()
    fields = set(re.findall(r"UGKWP_GAS_STATE_FIELD\([^,]+,\s*(\w+)\)", manifest))
    capabilities={"gasSpecies","gasThermalConductivity","gasGeometry","gasBoundaryLayer","gasBoundaryLayerModel",
                  "diagnostics","gasBoundaryProgram","gasSteadyMode","gravityX","gravityY","gravityZ"}
    indirect={"gasMu","riemannBoundaryPWaveLInf","riemannBoundaryPWaveGamma","riemannBoundaryPWaveFieldInf"}
    assert fields | capabilities == actual | indirect
    assert not any(re.search(r"particle|pool|packing|csr|cuda|Stream|Graph", f, re.I) for f in fields)

def test_applications_share_the_same_operator_owners():
    gas=ROOT/"applications/gasUGKP/private_backend/GpuResidentStrict.cu"
    thermal=ROOT/"common/thermal/GpuResidentStrict.cu"
    consumers=((ROOT/"applications/CHT/gpu/GpuResidentStrict.cu",thermal),
               (ROOT/"applications/gasUGKPSteady/private_backend/GpuSteadyBackend.cu",gas))
    def includes(path):
        return {(path.parent/name).resolve() for name in re.findall(r'^#include "([^"]+)"',path.read_text(),re.M)}
    for path,target in consumers:
        assert target.resolve() in includes(path)
    for path in (gas,thermal):
        source=path.read_text()
        assert (ROOT/"common/GpuGasAdvance.cuh").resolve() in includes(path)
        for header in GAS_HEADERS:
            assert f'#include "operators/{header}.cuh"' in source,(path,header)
            assert not re.search(r"__(?:global|device)__[^;{}]*\b"+header+r"\s*\(",source),(path,header)
