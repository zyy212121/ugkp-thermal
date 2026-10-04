                                                                   

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_split_behavior_is_derived_from_the_public_level_and_reported():
    wrapper = read("gpu/GpuResidentStrict.H")
    scheduling = read("../../common/GpuSchedulingConfiguration.H")
    assert '"gpuCsrSplitPreDirectory"' in scheduling
    assert "csrSplitPreDirectoryEnabled" in wrapper
    assert "value.splitPreDirectory = value.csrLevel != GpuCsrLevel::L0" in scheduling
    assert "has been removed" in scheduling
    assert '" splitPreDirectory="' in wrapper


def test_switch_crosses_every_interface_layer_without_growing_create_payload():
    for relative in (
        "gpu/GpuBackendApi.H",
        "gpu/GpuBackendProtocol.H",
        "gpu/GpuBackendClient.C",
        "private_backend/GpuBackendServer.C",
        "private_backend/GpuResidentStrict.cu",
    ):
        assert "csrSplitPreDirectoryEnabled" in read(relative), relative
    protocol = read("gpu/GpuBackendProtocol.H")
    assert "protocolMajor = 7" in protocol
    assert "protocolMinor = 0" in protocol
    assert "sizeof(CreateArgs) == 392" in protocol
    assert "dragReserved" not in protocol


def test_backend_validates_boolean_and_full_mode_forces_complete_dpre_rebuild():
    client = read("gpu/GpuBackendClient.C")
    server = read("private_backend/GpuBackendServer.C")
    cuda = read("private_backend/GpuResidentStrict.cu")
    validation = (
        "csrSplitPreDirectoryEnabled != 0"
        " && csrSplitPreDirectoryEnabled != 1"
    )
    assert validation in client
    assert "a.csrSplitPreDirectoryEnabled != 0" in server
    assert validation in cuda
    assert "s->csrSplitPreDirectoryEnabled = csrSplitPreDirectoryEnabled;" in cuda
    common = read("../../common/GpuParticleDirectoryHost.cuh")
    policy = read("../../common/GpuParticleDirectoryHostPolicy.cuh")
    prepare = common.index("int preparePreTransportParticleDirectory")
    full = common.index("return binParticlesByCell(s, block);", prepare)
    split_work = common.index("const int cellGrid", full)
    assert "!splitParticleDirectoryEnabled(s)" in common[prepare:full]
    assert "s->preBaseDirectoryReady == 0" in common[prepare:full]
    assert "return s->csrSplitPreDirectoryEnabled != 0;" in policy
    assert full < split_work
