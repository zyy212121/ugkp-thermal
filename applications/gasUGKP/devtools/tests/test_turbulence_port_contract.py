from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_four_turbulence_modes_and_openfoam_sst_semantics() -> None:
    configuration = read("readGpuGasConfiguration.H")
    assert 'fvSolutionDict.found("UGKP")' in configuration
    for token in (
        'simulationType == "laminar"',
        'gasTurbulenceModelName == "WALE"',
        'gasTurbulenceModelName == "Smagorinsky"',
        'simulationType == "RAS"',
        'gasTurbulenceModelName == "kOmegaSST"',
        "gasTurbulenceModel = 3",
        'lookupOrDefault<word>("wallTreatment", "lowRe")',
        'sstWallTreatmentName == "wallFunction"',
    ):
        assert token in configuration


def test_explicit_gpu_sst_path_is_available_to_pure_gas_and_two_phase() -> None:
    main = read("diluteUgkwpFoam.C")
    fields = read("createFields.H")
    cuda = read("private_backend/GpuResidentStrict.cu")
    protocol = read("gpu/GpuBackendProtocol.H")
    client = read("gpu/GpuBackendClient.C")
    server = read("private_backend/GpuBackendServer.C")
    assert main.count("configureResidentSst(resident);") == 2
    assert main.count("downloadSstToHostMirror") == 2
    assert 'wallDist::New(mesh).y()' in main
    assert 'IOobject("k"' in fields and 'IOobject("omega"' in fields
    for token in (
        "computeSstGradientsKernel",
        "computeSstFaceFluxKernel",
        "applySstFluxAndSourceKernel",
        "computeSstStabilityNumberKernel",
        "recoverSstPrimitivesKernel",
        "applySstWallFunctionStateKernel",
        "jayatillekeWallHeatFluxPrecomputed",
    ):
        assert token in cuda
    assert "kOmegaSST = 3" in protocol
    assert "SstConfigArgs" in protocol
    assert "Op::configureSst" in client + server
    assert "Op::downloadSst" in client + server
    for forbidden in ("fvm::", "solve(omega", "solve(k"):
        assert forbidden not in cuda


def test_new_binary_names_and_separated_link_boundary() -> None:
    assert "$(FOAM_USER_APPBIN)/gasUGKP" in read("Make/files")
    assert "GAS_UGKP_CUDA_BACKEND" in read("gpu/GpuBackendClient.C")
    build = read("private_backend/build_private_backend.sh")
    assert "gasUGKPCudaBackend" in build
    assert "GAS_UGKP_FMAD" in build
    assert "-lmomentumTransportModels" in read("Make/options")
    assert "static constexpr std::uint32_t magic = 0x47554750U;" in read(
        "gpu/GpuBackendProtocol.H"
    )


def test_non_fsh_branch_does_not_gain_fsh_runtime_state() -> None:
    runtime = "\n".join(
        read(path)
        for path in (
            "diluteUgkwpFoam.C",
            "gpu/GpuBackendApi.H",
            "gpu/GpuBackendProtocol.H",
            "gpu/GpuResidentStrict.H",
            "private_backend/GpuResidentStrict.cu",
        )
    )
    for forbidden in (
        "GpuCapillaryDetachment",
        "GpuSommerfeldSticking",
        "GpuParticleWallContactHeat",
        "pStuckFaceId",
        "particleWallDepositedEnergy",
    ):
        assert forbidden not in runtime
