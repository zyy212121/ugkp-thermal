pass








import os
from pathlib import Path
import shutil
import subprocess

import pytest

from validation.tools.paths import SOURCE_ROOT


@pytest.fixture(scope="module", params=[32, 64])
def sparse_executables(request, tmp_path_factory):
    if os.environ.get("UGKP_RUN_SPARSE_BLOCK8_CUDA") != "1":
        pytest.skip("set UGKP_RUN_SPARSE_BLOCK8_CUDA=1; requires exclusive GPU use")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA toolkit unavailable")
    bits = request.param
    source = Path(os.environ.get(
        f"UGKP_SPARSE_FP{bits}_SOURCE",
        str(SOURCE_ROOT / f".build/precision/fp{bits}/source"),
    )).expanduser().resolve()
    assert (source / "common/gpu/SparseConjugateGradient.cu").is_file(), (
        f"Materialize FP{bits} source before this opt-in test: {source}"
    )
    output = tmp_path_factory.mktemp(f"sparse-block8-{bits}")
    binaries = {}
    for fixture in ("sparse_block8_api", "sparse_block8_embedded_graph"):
        executable = output / fixture
        command = [
            nvcc, "--Ofast-compile=min", "-std=c++17", "-O2", "--fmad=false",
            "--expt-relaxed-constexpr", "-arch=" + os.environ.get("UGKP_CUDA_ARCH", "sm_89"),
            "-I" + str(source / "common"), "-I" + str(source / "common/gpu"),
            str(SOURCE_ROOT / f"validation/tests/{fixture}.cu"), "-o", str(executable),
        ]
        built = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (output / (fixture + "-build.log")).write_text(built.stdout + built.stderr)
        assert built.returncode == 0, built.stdout + built.stderr
        binaries[fixture] = executable
    return binaries


@pytest.mark.parametrize("fixture", ["sparse_block8_api", "sparse_block8_embedded_graph"])
@pytest.mark.parametrize("width", [128, 96])
@pytest.mark.parametrize("choice", [0, 8])
def test_sparse_optional_block8(sparse_executables, fixture, width, choice):
    executable = sparse_executables[fixture]
    result = subprocess.run(
        [str(executable), str(width), str(choice)],
        capture_output=True, text=True, timeout=300,
    )
    transcript = result.stdout + result.stderr
    (executable.parent / f"{fixture}-B{width}-P{choice}.log").write_text(transcript)
    print(transcript, end="")
    assert result.returncode == 0, transcript
    assert "PASS" in transcript
