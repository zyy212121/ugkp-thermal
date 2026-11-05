"""Host regression of CHMT adapters consuming the shared gas thermo."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

APP = (SOURCE_ROOT / 'applications/CHMT')

def test_shared_thermo_host(tmp_path):
    binary = tmp_path / "shared_thermo"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-I" + str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'), "-I" + str(APP.parents[1] / "common"),
                    str((VALIDATION_ROOT / 'solvers/CHMT/tests/test_shared_thermo.cpp')),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
