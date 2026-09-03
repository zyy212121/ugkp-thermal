#!/usr/bin/env python3
                                                   

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="strict")


def test_identity_and_direct_link_architecture_are_isolated() -> None:
    make_files = read("Make/files")
    build = read("tools/build_cuda_solver.sh")
    main = read("diluteUgkwpFoam.C")
    assert "$(FOAM_USER_APPBIN)/CHT" in make_files
    assert "libugkpcht_cuda_internal.a" in build
    assert "CHT" in main
    combined = make_files + build + main
    assert "GpuBackendServer" not in combined
    assert "gasUGKPCudaBackend" not in combined


def test_cold_wall_1d_wall_energy_reuses_the_accepted_enthalpy_flux() -> None:
    model = read("../../common/wall/GpuColdWallSolidification.H")
    assert "double acceptedWallPower = 0.0;" in model
    assert "acceptedWallPower = wallPower;" in model
    assert "const double wallEnergy = acceptedWallPower*deltaTSeconds;" in model
    assert "(guessedTemperature[0] - wallTemperatureK)*deltaTSeconds" not in model


def test_version_neutral_controls_and_weight_configuration_exist() -> None:
    config = read("readGpuGasConfiguration.H")
    fields = read("createFields.H")
    wrapper = read("gpu/GpuResidentStrict.H")
    assert 'fvSolutionDict.found("UGKP")' in config
    for name in ("parcelMass", "injectionParcelMass", "legacyRestartParcelMass"):
        assert f'"{name}"' in fields
    assert "injectionParcelMass" in wrapper
    assert "legacyRestartParcelMass_" in wrapper


def test_gravity_drag_and_block_size_cross_the_direct_link_abi() -> None:
    fields = read("createFields.H")
    wrapper = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    assert "uniformDimensionedVectorField g" in fields
    assert 'IOobject::READ_IF_PRESENT' in fields
    for name in ("SchillerNaumann", "GidaspowErgunWenYu"):
        assert name in read("gpu/GpuDragModel.H")
        assert name in read("gpu/GpuDragModels.cuh")
    for token in (
        "particleBlockThreads", "reductionBlockThreads",
        "dragModelId", "dragResidualRe",
        "gravityX", "gravityY", "gravityZ",
    ):
        assert token in wrapper
        assert token in cuda
    assert "applyEulerianGasSolidDragKernelStatic" in cuda
    assert "relaxMobileParticlesToResidentGasKernelStatic" in cuda
    assert "relaxWallBoundParticlesToResidentGasKernelStatic" in cuda
    assert "applyGasGravityKernel" in cuda


def test_dynamic_heavy_split_dpre_and_safe_warp_reduction_are_present() -> None:
    wrapper = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    scheduling = read("../../common/GpuSchedulingConfiguration.H")
    assert '"gpuParticleBlockThreads"' in scheduling
    assert '"gpuReductionBlockThreads"' in scheduling
    assert '"gpuCsrHeavyReductionAutoInterval"' in scheduling
    for token in (
        "preBaseCellOffset", "preBaseDirectoryReady",
        "buildSplitPreDirectory", "splitPreDirectoryActive",
        "updateDynamicHeavyPolicy", "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "dynamicHeavyThreshold", "csrHeavyTileParticles",
    ):
        assert token in cuda
    for obsolete in (
        "gpuCsrHeavyCellThreshold", "gpuCsrHeavyTileParticles",
        "gpuCsrHeavyWorkerBlocksPerSM",
    ):
        assert obsolete not in wrapper
    assert "constexpr unsigned int fullWarpMask = 0xffffffffu" in cuda
    assert "__activemask();\n    const int activeLanes" not in cuda


def test_weighted_restart_persists_mass_through_ordinary_and_thermal_writes() -> None:
    wrapper = read("gpu/GpuResidentStrict.H")
    restart_wrapper = read("gpu/GpuFshParticleRestart.H")
    restart = read("../FSH/gpu/GpuFshParticleRestart.H")
    thermal = read("thermal/GpuThermalExchangeState.C")
    cuda = read("gpu/GpuResidentStrict.cu")
    assert '../../FSH/gpu/GpuFshParticleRestart.H' in restart_wrapper
    assert 'formatName = "UGKP_FSH_PARTICLES_SCHEMA5_BIN"' in restart
    assert 'schema4FormatName = "UGKP_FSH_PARTICLES_SCHEMA4_BIN"' in restart
    assert 'schema3FormatName = "UGKP_FSH_PARTICLES_SCHEMA3_BIN"' in restart
    assert 'schema2FormatName = "UGKP_FSH_PARTICLES_SCHEMA2_BIN"' in restart
    assert "writeArray(os, v.pm, begin, chunk);" in restart
    assert "readArray(is, v.pm, begin, chunk);" in restart
    assert "pStuckFaceId" in restart
    assert "pContactMaximumArea" in restart
    assert "pContactPeakFraction" in restart
    assert "pColdNodeSpecificEnthalpy" in restart
    assert "pColdRingSolidMass" in restart
    assert "pCold2DNodeSpecificEnthalpy" in restart
    assert "pCold2DRingContactAge" in restart
    assert "pCold2DFrozenArea" in restart
    assert "legacyRestartParcelMass_" in wrapper
    assert "UGKP_PARTICLES_SCHEMA1_BIN" in thermal
    assert "UGKP_PARTICLES_SCHEMA4" in thermal
    for schema in range(1, 6):
        assert f"UGKP_FSH_PARTICLES_SCHEMA{schema}_BIN" in thermal
    assert "if (fshBinarySchema >= 4) scalar32Fields += 18;" in thermal
    assert "if (fshBinarySchema >= 5) scalar32Fields += 73;" in thermal
    assert "pm == nullptr" in cuda
    assert "!std::isfinite(pm[i]) || pm[i] <= 0.0" in cuda
    assert not re.search(r"s\.pm\s*\[[^]]+\]\s*=\s*s\.injectionParcelMass", cuda)


def test_cold_wall_deposit_restart_retains_consistent_contact_history() -> None:
    wrapper = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    for source in (wrapper, cuda):
        assert "pContactDuration[i] > 0.0f" in source or "s.pContactDuration[i] > 0.0f" in source
        assert "pContactPeakFraction[i] < 1.0f" in source or "s.pContactPeakFraction[i] < 1.0f" in source
    assert "pStuck[i] == 1 && pContactDuration[i] != 0.0f" not in wrapper
    assert "s.pContactDuration[i] != 0.0f" not in cuda[1500:90000]
    assert "!weightedFshV3 && !weightedFshV4 && !weightedFshV5" in wrapper


def test_cold_wall_2d_uses_the_shared_wall_directory_kernel() -> None:
    wrapper = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    model = read("../../common/wall/GpuColdWall2DSolidification.H")
    device = read("../../common/wall/GpuColdWall2DDevice.cuh")
    assert 'name == "coldWall1D"' in wrapper
    assert 'name == "coldWall2D"' in wrapper
    assert "coldWall2DNodeCount" in model
    assert "relaxColdWall2DParticlesToResidentGasKernel" in device
    assert "wallBoundDirectoryCount(s)" in device
    assert "particleWallColdWall2D" in device
    assert "if (s.coldWall2DEnabled != 0)" in cuda
    assert "GpuColdWall2DDevice.cuh" in cuda


def test_collision_correction_is_mass_weighted_and_retains_singleton_energy() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    for token in (
        "poolMass", "meanParticleMass", "sampleMeanDeltaX",
        "sampleMeanDeltaY", "sampleMeanDeltaZ", "targetTheta",
        "s.pTheta[i] = targetTheta",
    ):
        assert token in cuda
    assert "poolMass*sampleMeanDelta2" in cuda


def test_serial_translational_cyclic_maps_all_resident_subsystems() -> None:
    wrapper = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    boundary = read("gpu/GpuGasBoundarySpec.H")
    assert '#include "cyclicPolyPatch.H"' in wrapper
    assert "isA<cyclicPolyPatch>" in wrapper
    assert "facePeriodicPair" in wrapper
    assert "serial cyclic support" in wrapper
    assert "cyclic = 5" in boundary
    for token in (
        "isPeriodicFace", "periodicMappedCellCentre",
        "enforcePeriodicGasFluxAntisymmetryKernel",
        "enforcePeriodicSstFluxAntisymmetryKernel",
        "s.facePeriodicDx[faceI]", "kind == 5",
    ):
        assert token in cuda


def test_cht_thermal_ledgers_and_transactions_remain_in_place() -> None:
    main = read("diluteUgkwpFoam.C")
    header = read("gpu/GpuResidentStrict.H")
    cuda = read("gpu/GpuResidentStrict.cu")
    thermal_state = read("thermal/GpuThermalExchangeState.C")
    combined = main + header + cuda + thermal_state
    for token in (
        "GpuSolidThermalCoupler", "ThermalExchangeRestartState",
        "gasWallEnergy", "particleWallDepositedEnergy",
        "particleWallReflectedEnergy",
        "ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature",
        "ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums",
        "ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea",
        "ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked",
        "ugkwpGpuResidentStrictPeekGasWallEnergy",
        "ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy",
        "ugkwpGpuResidentStrictPeekParticleWallHeatLedgers",
        "ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers",
    ):
        assert token in combined
    assert "ledgerDt*s.gasPhiRhoE[f]" in cuda
    assert "accumulateParticleMomentsSegmentedKernel" in cuda
    for required in ("pStuckFaceId", "pDepositionArea", "lumpedParticleWallInterfaceContact"):
        assert required in combined


def test_generic_offline_refinement_tools_are_shipped_without_fsh_state() -> None:
    restart = read("tools/ugkp_particle_restart.py")
    refine = read("tools/refine_particle_cells.py")
    for token in ("UGKP_PARTICLES_SCHEMA1_BIN", "pm", "legacy_parcel_mass"):
        assert token in restart
    for token in ("mass", "mom_x", "mom_y", "mom_z", "energy", "mass_d", "mass_T"):
        assert token in refine
    assert "pStuck" not in restart + refine
    assert "pDepositionArea" not in restart + refine
