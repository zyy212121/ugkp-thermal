from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mixed_divergence_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("mixed_divergence") / "mixed_divergence_ale"
    source = ROOT / "validation/pr43/mixed_divergence_ale.cpp"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I"+str(ROOT),
                    "-I"+str(ROOT/"common"), "-I"+str(ROOT/"common/gasNumerics"),
                    str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize("case", [
    "static_zero", "moving_zero", "small_motion",
    "static_dt_volume", "moving_dt_volume", "subfloat_dt_volume",
])
def test_mixed_divergence_preserves_double_geometry_energy(mixed_divergence_binary, case):
    subprocess.run([str(mixed_divergence_binary), case], check=True, timeout=30)
