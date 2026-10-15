from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import subprocess
import tempfile
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/FSH')


def test_cold_wall_2d_geometry_and_phase_algebra():
    source = (VALIDATION_ROOT / 'solvers/FSH/tests/cold_wall_2d_algebra_test.cpp')
    with tempfile.TemporaryDirectory(prefix="fsh-cold-wall-2d-") as temporary:
        executable = Path(temporary) / "cold_wall_2d_algebra_test"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-I",
                str(ROOT / "../../common/wall"),
                str(source),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)
