from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


import subprocess
import tempfile
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/CHT')
SOURCE = (VALIDATION_ROOT / 'solvers/CHT/devtools/tests/sst_algebra_test.cpp')


def test_host_cuda_shared_sst_algebra_matches_openfoam10_equations() -> None:
    with tempfile.TemporaryDirectory(prefix="cht-sst-") as temporary:
        executable = Path(temporary) / "sst_algebra_test"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-pedantic",
                str(SOURCE),
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(executable)], check=True)
