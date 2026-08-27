from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_ugkp_has_an_independent_runtime_identity() -> None:
    make = read("Make/files")
    public_build = read("tools/build_public_frontend.sh")
    private_build = read("private_backend/build_private_backend.sh")
    client = read("gpu/GpuBackendClient.C")
    protocol = read("gpu/GpuBackendProtocol.H")
    solver = read("diluteUgkwpFoam.C")

    assert "$(FOAM_USER_APPBIN)/gasUGKP" in make
    assert 'frontend="${FOAM_USER_APPBIN}/gasUGKP"' in public_build
    assert 'frontend_bin="${FOAM_USER_APPBIN}/gasUGKP"' in private_build
    assert 'backend_bin="${FOAM_USER_APPBIN}/gasUGKPCudaBackend"' in private_build
    assert "GAS_UGKP_FMAD" in private_build
    assert 'std::getenv("GAS_UGKP_CUDA_BACKEND")' in client
    assert client.count('"gasUGKPCudaBackend"') == 2
    assert "magic = 0x47554750U;" in protocol
    assert "protocolMajor = 7" in protocol
    assert "gasUGKP:" in solver

    active = "\n".join((make, public_build, private_build, client, protocol, solver))
    for inherited in (
        "ugkp.95",
        "ugkp5CudaBackend",
        '"UGKP_CUDA_BACKEND"',
        '"UGKP_FMAD"',
        '"G295"',
    ):
        assert inherited not in active
