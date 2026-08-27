from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.skip(
    reason=(
        "historical UGKP/2.8 source-fingerprint contract is superseded by "
        "the UGKPUnified shared wall-state and normalized configuration contract"
    )
)


ROOT = Path(__file__).resolve().parents[2]
CHT_FROZEN_CASE = ROOT.parent / "UGKP" / "939insideCHT"
UGKP_GAS_AUTHORITY = ROOT.parent / "gasUGKP"
CHT_THERMAL_AUTHORITY = ROOT.parent / "UGKP" / "thermal"
CASE = ROOT / "939insideCHT"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_packaged_939insidecht() -> None:
    if not CASE.exists():
        pytest.skip(
            "939insideCHT is intentionally absent from the clean "
            "UGKP package"
        )
    assert CASE.is_dir(), "packaged 939insideCHT path is not a directory"


def test_integrated_backend_and_ugkpcht_gas_path() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    header = read("gpu/GpuResidentStrict.H")
    build = read("tools/build_cuda_solver.sh")
    make_files = read("Make/files")

    for token in (
        "CharacteristicMuscl.cuh",
        "OpenFoamLimitedLinear.cuh",
        "OpenFoamViscousFlux.cuh",
        "RiemannBoundaryState.cuh",
        "RiemannGasFlux.cuh",
        "computeRiemannGasFaceFluxDevice",
        "advanceGasEulerSubstage",
        "gpuCsrCellLocalPath",
        "GpuGasBoundarySpec.H",
    ):
        assert token in cuda + header

    assert "GpuBackendApi.H" not in cuda
    assert "GpuBackendServer" not in cuda + header + build
    assert "UGKP_DEVELOPMENT_PROBES" not in cuda
    assert "DevelopmentAdvanceProbe" not in cuda
    assert "libugkpcht_cuda_internal.a" in build
    assert "${FOAM_USER_APPBIN}/CHT" in build
    assert "$(FOAM_USER_APPBIN)/CHT" in make_files
    assert "libugkpcht_cuda_internal.a" not in build
    assert "diluteUgkwpFoam_UGKP_CHT" not in make_files + build


def test_ugkp_gas_authority_numerical_headers_are_exactly_shared() -> None:
    for name in (
        "CharacteristicMuscl.cuh",
        "OpenFoamLimitedLinear.cuh",
        "OpenFoamViscousFlux.cuh",
        "RiemannBoundaryState.cuh",
        "RiemannGasFlux.cuh",
    ):
        assert sha256(ROOT / "gpu" / name) == sha256(
            UGKP_GAS_AUTHORITY / "private_backend" / name
        )
    assert sha256(ROOT / "gpu/GpuGasBoundarySpec.H") == sha256(
        UGKP_GAS_AUTHORITY / "gpu/GpuGasBoundarySpec.H"
    )


def test_cht_impact_thermal_and_restart_contract_is_preserved() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    header = read("gpu/GpuResidentStrict.H")
    main = read("diluteUgkwpFoam.C")
    thermal_state = read("thermal/GpuThermalExchangeState.C")

    for token in (
        "gasWallEnergy",
        "particleWallEnergy",
        "particleWallContactMask",
        "ugkwpGpuResidentStrictConfigureGasWallEnergyLedger",
        "ugkwpGpuResidentStrictConfigureParticleWallHeatTransfer",
        "ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature",
        "ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums",
        "ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea",
        "ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked",
        "UGKP_PARTICLES_SCHEMA4",
        "UGKP_PARTICLES_SCHEMA1_BIN",
        "thermalExchangeState",
    ):
        assert token in cuda + header + main + thermal_state

    for forbidden in (
        "pStuckFaceId",
        "pAdep",
        "particleWallDepositionMask",
        "particleWallContactAreaScale",
        "depositedParticleWallContact",
        "UGKP_PARTICLES_SCHEMA3",
    ):
        assert forbidden not in cuda + header + thermal_state

    thermal_refresh = cuda.split(
        'extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked', 1
    )[1].split('extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev', 1)[0]
    assert "refreshPackedParticleEnthalpyKernel" in thermal_refresh
    assert "accumulateParticleEnthalpyMomentAtomicKernel" in thermal_refresh
    assert "binParticlesByCell" not in thermal_refresh
    assert "initialiseEpsGPrevKernel" not in thermal_refresh
    assert "rebuildResidentParticleMomentsFromParticles" not in thermal_refresh
    assert "dt*ledgerWeight" in cuda
    assert "s->hostGasTimeIntegrator == 3 ? 1.0/6.0 : 0.5" in cuda
    assert "advanceGasEulerSubstage(s, dt, 2.0/3.0)" in cuda


def test_l0_compaction_carries_mobile_particle_metadata_only() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    gather = re.search(
        r"__global__ void gatherSelectedParticlesKernel.*?"
        r"template<class T>",
        cuda,
        re.DOTALL,
    )
    assert gather
    assert "compactPT[dst] = s.pT[i]" in gather.group()
    assert "compactPTheta[dst] = s.pTheta[i]" in gather.group()
    assert "compactPd[dst] = s.pd[i]" in gather.group()
    assert "pStuckFaceId" not in gather.group()
    assert "pAdep" not in gather.group()

    selected_commit = re.search(
        r"__global__ void commitSelectedParticleBuffersKernel.*?"
        r"template<class T>\s*void swapParticlePointerHost",
        cuda,
        re.DOTALL,
    )
    assert selected_commit
    assert "s.pT, s.compactPT" in selected_commit.group()
    assert "s.pTheta, s.compactPTheta" in selected_commit.group()
    assert "pStuckFaceId" not in selected_commit.group()
    assert "pAdep" not in selected_commit.group()


def test_ugkpcht_thermal_orchestration_changes_only_weighted_restart_preflight() -> None:
    migrated_root = ROOT / "thermal"
    authority_files = {
        path.relative_to(CHT_THERMAL_AUTHORITY)
        for path in CHT_THERMAL_AUTHORITY.rglob("*")
        if path.suffix in {".C", ".H"}
    }
    migrated_files = {
        path.relative_to(migrated_root)
        for path in migrated_root.rglob("*")
        if path.suffix in {".C", ".H"}
    }
    assert authority_files, "UGKP thermal authority is empty"
    assert migrated_files == authority_files
    changed = {Path("GpuThermalExchangeState.C")}
    for relative in sorted(authority_files - changed):
        assert sha256(CHT_THERMAL_AUTHORITY / relative) == sha256(
            migrated_root / relative
        ), relative
    state = read("thermal/GpuThermalExchangeState.C")
    assert "UGKP_PARTICLES_SCHEMA1_BIN" in state
    assert "UGKP_PARTICLES_SCHEMA4" in state


def test_939insidecht_native_l0_acceptance_configuration() -> None:
    require_packaged_939insidecht()
    props = (CASE / "constant/ugkwpProperties").read_text(encoding="utf-8")
    schemes = (CASE / "system/fluid/fvSchemes").read_text(encoding="utf-8")
    solution = (CASE / "system/fluid/fvSolution").read_text(encoding="utf-8")
    control = (CASE / "system/controlDict").read_text(encoding="utf-8")
    physical = (
        CASE / "constant/fluid/physicalProperties"
    ).read_text(encoding="utf-8")

    assert "gpuCsrCellLocalPath false;" in props
    assert "gpuCsrHeavyReduction false;" in props
    assert "gpuCsrWarpAggregatedBinning false;" in props
    assert "fluxScheme      SLAU2.2;" in schemes
    assert "Gauss linear corrected" in schemes
    assert "UGKP" in solution
    assert "UGKP" not in solution
    assert "startFrom       startTime;" in control
    assert "startTime       0.1;" in control
    assert "endTime         0.2;" in control
    assert re.search(r"\bapplication\s+ugkp\.8chtsst\s*;", control)
    assert "diluteUgkwpFoam_UGKP_CHT" not in control
    assert "perfectGas" in physical
    assert not (CASE / "constant/fluid/gksProperties").exists()
    directory_names = {
        path.name for path in CASE.iterdir() if path.is_dir()
    }
    assert {"0", "0.1", "constant", "system"} <= directory_names
    result_directories = directory_names - {
        "0",
        "0.1",
        "constant",
        "system",
    }
    assert len(result_directories) <= 1
    if result_directories:
        result_time = float(next(iter(result_directories)))
        assert abs(result_time - 0.2) <= 2.0e-4

    for time_name in ("0", "0.1"):
        velocity = (CASE / time_name / "fluid/U").read_text(encoding="utf-8")
        assert velocity.count("type            noSlip;") + velocity.count(
            "type noSlip;"
        ) == 4


def test_939insidecht_frozen_start_fields_only_change_wall_u_semantics() -> None:
    require_packaged_939insidecht()
    baseline_case = CHT_FROZEN_CASE
    for time_name in ("0", "0.1"):
        baseline_root = baseline_case / time_name
        migrated_root = CASE / time_name
        baseline_files = {
            path.relative_to(baseline_root)
            for path in baseline_root.rglob("*")
            if path.is_file()
        }
        migrated_files = {
            path.relative_to(migrated_root)
            for path in migrated_root.rglob("*")
            if path.is_file()
        }
        assert migrated_files == baseline_files
        for relative in baseline_files:
            baseline = baseline_root / relative
            migrated = migrated_root / relative
            if relative == Path("fluid/U"):
                migrated_text = migrated.read_text(encoding="utf-8")
                assert (
                    migrated_text.replace(
                        "type            noSlip;",
                        "type            zeroGradient;",
                    ).replace(
                        "type noSlip;",
                        "type zeroGradient;",
                    )
                    == baseline.read_text(encoding="utf-8")
                )
            elif relative == Path("fluid/rho"):
                migrated_text = migrated.read_text(encoding="utf-8")
                assert (
                    migrated_text.replace(
                        "type            calculated;",
                        "type            fixedValue;",
                    ).replace(
                        "type calculated;",
                        "type fixedValue;",
                    )
                    == baseline.read_text(encoding="utf-8")
                )
            elif (
                time_name == "0.1"
                and relative == Path("thermalExchangeState")
            ):
                migrated_text = migrated.read_text(encoding="utf-8")
                assert (
                    migrated_text.replace(
                        "a5f72e6be4797e0aaa1047e9ee727c59af4f825b",
                        "427765f48cee61d1ee34ac2a889073c267a29c59",
                    )
                    == baseline.read_text(encoding="utf-8")
                )
            else:
                assert sha256(migrated) == sha256(baseline), (
                    time_name,
                    relative,
                )
