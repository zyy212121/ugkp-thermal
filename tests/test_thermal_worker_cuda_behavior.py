from pathlib import Path
import os, shutil, subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/thermal_workers"

def function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]

def compile_and_run(tmp_path, text, bits):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(nvcc).is_file() or not Path(smi).is_file():
        pytest.skip("CUDA compiler and GPU required")
    if subprocess.run([smi, "-L"], capture_output=True).returncode:
        pytest.skip("CUDA GPU unavailable")
    source, binary = tmp_path / "behavior.cu", tmp_path / "behavior"
    source.write_text(text)
    command = [nvcc, "-O3", "-std=c++17", "--fmad=false",
               "-arch=" + os.environ.get("UGKWP_CUDA_ARCH", "sm_89"),
               "-DUGKWP_GPU_REAL_BITS=" + str(bits),
               "-I" + str(ROOT / "common"),
               "-I" + str(ROOT / "gpu/thermal"), str(source), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)

@pytest.mark.parametrize("branch,bits", [("FSH", 64), ("CHT", 32), ("CHT", 64)])
def test_gather_preserves_all_payload_order_and_exact_wall_publication(tmp_path, branch, bits):
    folder = "private_backend" if branch == "FSH" else "gpu"
    source = (ROOT / "applications" / branch / folder / "GpuResidentStrict.cu").read_text()
    hook = function(source, "__device__ __forceinline__ void copyCellLocalParticle")
    fixture = (FIXTURES / ("gather_" + branch + ".cu.in")).read_text()
    compile_and_run(tmp_path, fixture.replace("@PARTICLE_COPY_HOOK@", hook)
                    .replace("@REAL_TYPE@", "float" if bits == 32 else "double"), bits)

@pytest.mark.parametrize("bits", [32, 64])
def test_shared_persistent_dispatch_matches_independent_sums_and_legacy(tmp_path, bits):
    # Physical accumulation is replaced with independently known exact sums;
    # both production dispatch headers and their launch helpers run on the GPU.
    compile_and_run(tmp_path, (FIXTURES / "direct_dispatch.cu").read_text(), bits)
