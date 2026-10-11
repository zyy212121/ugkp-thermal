"""Host regression of CHMT adapters consuming the shared gas thermo."""
from pathlib import Path
import subprocess

APP = Path(__file__).resolve().parents[1]

def test_shared_thermo_host(tmp_path):
    binary = tmp_path / "shared_thermo"
    subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I" + str(APP), "-I" + str(APP.parents[1] / "common"),
                    str(APP / "tests/test_shared_thermo.cpp"),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
