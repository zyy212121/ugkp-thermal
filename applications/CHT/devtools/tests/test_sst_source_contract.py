from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_ugkpcht_build_artifacts_are_isolated_from_older_cht() -> None:
    make_files = read("Make/files")
    build = read("tools/build_cuda_solver.sh")

    assert "$(FOAM_USER_APPBIN)/CHT" in make_files
    assert "libugkpcht_cuda_internal.a" in build
    assert 'UGKWP_CUDA_LOG_PHASE:-ugkpcht_upgrade' in build
    assert '${FOAM_USER_APPBIN}/CHT' in build

    assert "diluteUgkwpFoam_UGKP_CHT" not in make_files + build


def test_ugkpcht_runtime_identifies_the_new_solver() -> None:
    main = read("diluteUgkwpFoam.C")
    assert "CHT" in main
    assert "UGKP requires" not in main


def test_openfoam_ras_sst_configuration_is_supported() -> None:
    configuration = read("readGpuGasConfiguration.H")
    for token in (
        'simulationType == "RAS"',
        'gasTurbulenceModelName == "kOmegaSST"',
        "gasTurbulenceModel = 3",
        'subDict("RAS")',
        'subOrEmptyDict("kOmegaSSTCoeffs")',
        'lookupOrDefault<scalar>("alphaK1", scalar(0.85))',
        'lookupOrDefault<scalar>("alphaOmega2", scalar(0.856))',
        'lookupOrDefault<scalar>("betaStar", scalar(0.09))',
        'lookupOrDefault<scalar>("gamma1", scalar(5.0/9.0))',
        'lookupOrDefault<scalar>("a1", scalar(0.31))',
        'lookupOrDefault<scalar>("kMin", scalar(1.0e-12))',
        'lookupOrDefault<scalar>("omegaMin", scalar(1.0e-6))',
        'lookupOrDefault<scalar>("maxSourceNumber", scalar(0.25))',
    ):
        assert token in configuration

    assert 'simulationType == "RANS"' not in configuration


def test_sst_restart_fields_and_static_wall_distance_are_created() -> None:
    fields = read("createFields.H")
    main = read("diluteUgkwpFoam.C")
    assert 'IOobject("k"' in fields
    assert 'IOobject("omega"' in fields
    assert "volScalarField nut" in fields
    assert '"nut"' in fields
    assert 'wallDist::New(mesh).y()' in main


def test_sst_runtime_path_is_explicit_and_gpu_resident() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    header = read("gpu/GpuResidentStrict.H")
    combined = cuda + header
    for token in (
        "rhoK",
        "rhoOmega",
        "computeSstGradientsKernel",
        "computeSstFaceFluxKernel",
        "applySstFluxAndSourceKernel",
        "computeSstStabilityNumberKernel",
        "recoverSstPrimitivesKernel",
        "sstLowReWallOmega",
        "sstMaxSourceNumber",
        "ugkwpGpuResidentStrictConfigureSst",
        "ugkwpGpuResidentStrictDownloadSst",
    ):
        assert token in combined

    for forbidden in (
        "fvm::ddt",
        "fvm::laplacian",
        "solve(omega",
        "solve(k",
        "localEulerDdt",
        "analyticSstSource",
    ):
        assert forbidden not in combined


def test_sst_retains_selectable_resolved_low_re_wall() -> None:
    cuda = read("gpu/GpuResidentStrict.cu")
    assert "if (s.turbulenceModel == 3)" in cuda
    assert "s.sstWallTreatment == 0" in cuda
    assert "muTurbulent = 0.0;" in cuda
    assert "kTurbulent = 0.0;" in cuda
    assert "6.0*nu/(coefficients.beta1*ySafe*ySafe)" in read(
        "gpu/GpuSstAlgebra.cuh"
    )


def test_sst_smoke_uses_standard_openfoam_user_files() -> None:
    if not (ROOT / "examples/sst_explicit_gpu_smoke").is_dir():
        pytest.skip("minimal unified package omits the optional SST smoke case")
    momentum = read("examples/sst_explicit_gpu_smoke/constant/momentumTransport")
    k_field = read("examples/sst_explicit_gpu_smoke/0/k")
    omega_field = read("examples/sst_explicit_gpu_smoke/0/omega")
    make_options = read("Make/options")

    normalized_momentum = " ".join(momentum.split())
    assert "simulationType RAS;" in normalized_momentum
    assert "model kOmegaSST;" in normalized_momentum
    assert "kqRWallFunction" in k_field
    assert "omegaWallFunction" in omega_field
    assert "-lmomentumTransportModels" in make_options


def test_restart_acceptance_driver_checks_k_omega_and_nut() -> None:
    if not (ROOT / "examples/sst_explicit_gpu_smoke").is_dir():
        pytest.skip("minimal unified package omits the optional SST smoke case")
    restart = read("examples/sst_explicit_gpu_smoke/check_sst_restart.py")
    for token in (
        'FIELDS = ("k", "omega", "nut")',
        '"startFrom", "latestTime"',
        '"endTime", "4e-5"',
        "max_rel > 1.0e-10",
    ):
        assert token in restart


def test_sst_wall_function_configuration_and_resident_path() -> None:
    configuration = read("readGpuGasConfiguration.H")
    main = read("diluteUgkwpFoam.C")
    cuda = read("gpu/GpuResidentStrict.cu")
    header = read("gpu/GpuResidentStrict.H")
    wall = read("gpu/OpenFoamWallFunctions.cuh")
    combined = cuda + header

    for token in (
        'lookupOrDefault<word>("wallTreatment", "lowRe")',
        'lookupOrDefault<scalar>("wallKappa", scalar(0.41))',
        'lookupOrDefault<scalar>("wallE", scalar(9.8))',
        'lookupOrDefault<scalar>("wallCmu", scalar(0.09))',
        'sstWallTreatmentName == "wallFunction"',
    ):
        assert token in configuration

    for token in (
        "sstWallTreatment",
        "sstWallKappa",
        "sstWallE",
        "sstWallCmu",
        "applySstWallFunctionStateKernel",
        "omegaWallFunctionState",
        "jayatillekeWallHeatFlux",
        "jayatillekeWallHeatFluxPrecomputed",
        "sstJayatillekeP",
        "sstThermalYPlus",
        "directWallHeatFluxActive",
    ):
        assert token in combined + wall + main

    assert "wallTreatment=" in main
    assert "fvm::" not in combined
    assert "solve(omega" not in combined


def test_wall_function_gpu_smoke_case_is_self_checking() -> None:
    if not (ROOT / "examples/sst_wall_function_gpu_smoke").is_dir():
        pytest.skip("minimal unified package omits the optional wall-function smoke case")
    momentum = read(
        "examples/sst_wall_function_gpu_smoke/constant/momentumTransport"
    )
    runner = read("examples/sst_wall_function_gpu_smoke/Allrun")
    checker = read(
        "examples/sst_wall_function_gpu_smoke/check_wall_function_gpu.py"
    )
    assert "wallTreatment   wallFunction;" in momentum
    assert "check_wall_function_gpu.py" in runner
    assert '"wallTreatment=wallFunction"' in checker
    assert "mean_temperature >= 1200.0" in checker
