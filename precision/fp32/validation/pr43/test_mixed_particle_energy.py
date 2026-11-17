from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("fixture, thermal", [("mixed_particle_energy", 1),
                                             ("mixed_particle_energy", 0),
                                             ("mixed_coupling_energy", 1)])
def test_mixed_particle_authoritative_energy(tmp_path, fixture, thermal):
    source = ROOT / "validation/pr43" / (fixture + ".cpp")
    binary = tmp_path / fixture
    subprocess.run(["g++", "-std=c++17", "-O2", "-DFIXTURE_THERMAL="+str(thermal), "-I"+str(ROOT),
                    "-I"+str(ROOT/"common"), "-I"+str(ROOT/"common/gasNumerics"),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
