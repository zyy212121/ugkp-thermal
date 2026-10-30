"""Variable-transport equilibrium blowing, dimensional balances and failure status."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("probe", ["equilibrium_co2_wall.cu", "wall_matching_temperature.cu", "adiabatic_coherent.cpp"])
def test_equilibrium_co2_wall(tmp_path, probe):
    executable = tmp_path / "equilibrium-wall"
    subprocess.run([
        "g++", "-x", "c++", "-std=c++17", "-O2", "-I" + str(ROOT / "common"),
        str(ROOT / "validation/pr32" / probe), "-o", str(executable),
    ], check=True)
    subprocess.run([str(executable)], check=True)
