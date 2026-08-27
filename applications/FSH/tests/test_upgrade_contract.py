#!/usr/bin/env python3
                                                             

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="strict")


def test_fsh_has_an_isolated_frontend_backend_identity() -> None:
    assert "FSH" in source("Make/files")
    assert "FSHCudaBackend" in source("private_backend/build_private_backend.sh")
    assert "FSH_CUDA_BACKEND" in source("gpu/GpuBackendClient.C")
    protocol = source("gpu/GpuBackendProtocol.H")
    assert "magic = 0x46534842U;" in protocol
    assert "protocolMajor = 6" in protocol
    assert "protocolMinor = 0" in protocol


def test_version_neutral_case_controls_and_weight_configuration_exist() -> None:
    config = source("readGpuGasConfiguration.H")
    fields = source("createFields.H")
    wrapper = source("gpu/GpuResidentStrict.H")
    assert 'fvSolutionDict.found("UGKP")' in config
    for name in ("parcelMass", "injectionParcelMass", "legacyRestartParcelMass"):
        assert f'"{name}"' in fields
    assert "injectionParcelMass" in wrapper
    assert "legacyRestartParcelMass" in wrapper


def test_gravity_drag_and_configurable_block_cross_the_process_boundary() -> None:
    fields = source("createFields.H")
    assert "typeIOobject<uniformDimensionedVectorField> gravityHeader" in fields
    assert '"g",' in fields
    assert "runTime.constant()" in fields
    drag = source("gpu/GpuDragModel.H")
    device_drag = source("private_backend/GpuDragModels.cuh")
    assert "SchillerNaumann" in drag
    assert "GidaspowErgunWenYu" in drag
    assert "SchillerNaumannDrag" in device_drag
    assert "GidaspowErgunWenYuDrag" in device_drag
    paths = (
        "gpu/GpuBackendApi.H", "gpu/GpuBackendProtocol.H",
        "gpu/GpuBackendClient.C", "private_backend/GpuBackendServer.C",
        "gpu/GpuResidentStrict.H", "private_backend/GpuResidentStrict.cu",
    )
    for relative in paths:
        text = source(relative)
        for token in ("dragModelId", "gravityX", "gravityY", "gravityZ", "particleBlockThreads", "reductionBlockThreads"):
            assert token in text, f"{token} is missing from {relative}"


def test_dynamic_heavy_and_split_dpre_replace_fixed_heavy_controls() -> None:
    wrapper = source("gpu/GpuResidentStrict.H")
    cuda = source("private_backend/GpuResidentStrict.cu")
    scheduling = source("../../common/GpuSchedulingConfiguration.H")
    assert '"gpuParticleBlockThreads"' in scheduling
    assert '"gpuReductionBlockThreads"' in scheduling
    assert '"gpuCsrHeavyReductionAutoInterval"' in scheduling
    for token in ("preBaseCellOffset", "preBaseDirectoryReady", "buildSplitPreDirectory", "buildPostTransportDirectory", "dynamicHeavyThreshold"):
        assert token in cuda
    assert "occupancy" in cuda.lower()
    for old_name in ("gpuCsrHeavyCellThreshold", "gpuCsrHeavyTileParticles", "gpuCsrHeavyWorkerBlocksPerSM"):
        assert old_name not in wrapper


def test_weighted_restart_is_fsh_specific_and_preserves_stuck_state() -> None:
    wrapper = source("gpu/GpuResidentStrict.H")
    cuda = source("private_backend/GpuResidentStrict.cu")
    api = source("gpu/GpuBackendApi.H")
    restart = source("gpu/GpuFshParticleRestart.H")
    assert 'formatName = "UGKP_FSH_PARTICLES_SCHEMA5_BIN"' in restart
    assert 'schema4FormatName = "UGKP_FSH_PARTICLES_SCHEMA4_BIN"' in restart
    assert 'schema3FormatName = "UGKP_FSH_PARTICLES_SCHEMA3_BIN"' in restart
    assert 'schema2FormatName = "UGKP_FSH_PARTICLES_SCHEMA2_BIN"' in restart
    assert 'legacyFormatName = "UGKP_FSH_PARTICLES_SCHEMA1_BIN"' in restart
    assert "UGKP_PARTICLES_SCHEMA2" in wrapper
    assert "UGKP_PARTICLES_SCHEMA3" in wrapper
    assert "UGKP_PARTICLES_SCHEMA1_BIN" not in wrapper
    assert "UGKP_PARTICLES_SCHEMA1_BIN" not in restart
    for token in ("pm", "pStuck", "pStuckFaceId", "pDepositionArea", "pContactDuration", "pContactMaximumArea", "pContactPeakFraction", "pColdNodeSpecificEnthalpy", "pColdRingSolidMass", "pColdFrozenArea", "pColdContactAge", "pCold2DNodeSpecificEnthalpy", "pCold2DRingContactAge", "pCold2DFrozenArea"):
        assert token in api and token in wrapper and token in cuda
    assert "legacyRestartParcelMass" in wrapper
    assert not re.search(r"s\.pm\s*\[[^]]+\]\s*=\s*s\.parcelMass", cuda)


def test_cold_wall_2d_is_independent_and_wall_directory_scoped() -> None:
    wrapper = source("gpu/GpuResidentStrict.H")
    cuda = source("private_backend/GpuResidentStrict.cu")
    model = source("../../common/wall/GpuColdWall2DSolidification.H")
    device = source("../../common/wall/GpuColdWall2DDevice.cuh")
    wall_model = source("../../common/wall/GpuColdWallSolidification.H")
    assert "particleWallColdWall2D = 4" in wall_model
    assert 'name == "coldWall1D"' in wrapper
    assert 'name == "coldWall2D"' in wrapper
    assert "coldWall2DRadialNodeCount = 8" in model
    assert "coldWall2DAxialNodeCount = 8" in model
    assert "relaxColdWall2DParticlesToResidentGasKernel" in device
    assert "wallBoundDirectoryCount(s)" in device
    assert "wallBoundDirectoryParticle(s, entry)" in device
    assert "particleWallColdWall2D" in device
    assert "if (s.coldWall2DEnabled != 0)" in cuda
    assert "GpuColdWall2DDevice.cuh" in cuda


def test_cold_wall_1d_wall_energy_reuses_the_accepted_enthalpy_flux() -> None:
    model = source("../../common/wall/GpuColdWallSolidification.H")
    assert "double acceptedWallPower = 0.0;" in model
    assert "acceptedWallPower = wallPower;" in model
    assert "const double wallEnergy = acceptedWallPower*deltaTSeconds;" in model
    assert "(guessedTemperature[0] - wallTemperatureK)*deltaTSeconds" not in model


def test_collision_resampling_is_mass_weighted_and_keeps_singleton_energy() -> None:
    cuda = source("private_backend/GpuResidentStrict.cu")
    for token in ("sampleMass", "sampleFluctuationX", "sampleFluctuationEnergy", "targetRandomEnergy", "retainSingletonTheta"):
        assert token in cuda


def test_every_fsh_particle_lifecycle_keeps_deposition_state() -> None:
    cuda = source("private_backend/GpuResidentStrict.cu")
    for name in ("PStuck", "PStuckFaceId", "PDepositionArea"):
        assert f"compact{name}" in cuda
    assert "depositedWallEnergyJ" in cuda
    assert "reflectedWallEnergyJ" in cuda


def test_offline_fsh_refinement_tool_preserves_extended_state() -> None:
    restart_tool = source("tools/fsh_particle_restart.py")
    refine_tool = source("tools/refine_fsh_particle_cells.py")
    for token in ("pm", "pStuck", "pStuckFaceId", "pDepositionArea"):
        assert token in restart_tool and token in refine_tool
    for token in ("mass", "mom_x", "mom_y", "mom_z", "energy", "mass_d", "mass_T"):
        assert token in refine_tool


def test_serial_translational_cyclic_crosses_every_layer_without_reusing_stuck_kind() -> None:
    wrapper = source("gpu/GpuResidentStrict.H")
    cuda = source("private_backend/GpuResidentStrict.cu")
    assert "isA<cyclicPolyPatch>" in wrapper
    assert "facePeriodicPair" in wrapper
    assert "facePeriodicDx" in wrapper
    assert "kind = 6;" in wrapper
    assert "kind = 5;" in wrapper
    for token in (
        "facePeriodicPair", "facePeriodicDx", "facePeriodicDy", "facePeriodicDz",
        "periodicMappedCellCentre", "enforcePeriodicGasFluxAntisymmetryKernel",
        "enforcePeriodicSstFluxAntisymmetryKernel",
    ):
        assert token in cuda
