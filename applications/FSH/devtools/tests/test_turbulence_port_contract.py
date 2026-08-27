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


def test_explicit_gpu_sst_is_orthogonal_to_fsh_particle_physics() -> None:
    main = read("diluteUgkwpFoam.C")
    fields = read("createFields.H")
    cuda = read("private_backend/GpuResidentStrict.cu")
    protocol = read("gpu/GpuBackendProtocol.H")
    assert main.count("configureResidentSst(resident);") == 2
    assert main.count("downloadSstToHostMirror") == 2
    assert 'wallDist::New(mesh).y()' in main
    assert 'IOobject("k"' in fields and 'IOobject("omega"' in fields
    for token in (
        "computeSstGradientsKernel",
        "computeSstFaceFluxKernel",
        "applySstFluxAndSourceKernel",
        "applySstWallFunctionStateKernel",
        "jayatillekeWallHeatFluxPrecomputed",
    ):
        assert token in cuda
    for retained in (
        '#include "../thermal/GpuParticleWallContactHeat.H"',
        '#include "../thermal/GpuCapillaryDetachment.H"',
        '#include "../thermal/GpuSommerfeldSticking.H"',
        "particleWallDepositedEnergy",
        "particleWallReflectedEnergy",
        "pStuckFaceId",
        "pDepositionArea",
        "evaluateCapillaryDetachmentState",
    ):
        assert retained in cuda
    for header in (
        "thermal/GpuParticleWallContactHeat.H",
        "thermal/GpuCapillaryDetachment.H",
        "thermal/GpuSommerfeldSticking.H",
        "thermal/GpuAluminaProperties.H",
    ):
        assert (ROOT / header).is_file()
    assert "kOmegaSST = 3" in protocol and "SstConfigArgs" in protocol


def test_new_binary_names_and_separated_link_boundary() -> None:
    assert "$(FOAM_USER_APPBIN)/FSH" in read("Make/files")
    assert "FSH_CUDA_BACKEND" in read("gpu/GpuBackendClient.C")
    build = read("private_backend/build_private_backend.sh")
    assert "FSHCudaBackend" in build
    assert "FSH_FMAD" in build
    assert "-lmomentumTransportModels" in read("Make/options")
    assert "magic = 0x46534842U;" in read("gpu/GpuBackendProtocol.H")
