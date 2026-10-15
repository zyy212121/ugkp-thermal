from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


import subprocess
import tempfile
from pathlib import Path
import pytest


ROOT = (SOURCE_ROOT / 'applications/FSH')
SOURCE = (VALIDATION_ROOT / 'solvers/FSH/devtools/tests/sst_algebra_test.cpp')


def test_host_cuda_shared_sst_algebra_matches_openfoam10_equations() -> None:
    if not SOURCE.is_file():
        pytest.skip("optional standalone SST algebra source is not packaged")
    with tempfile.TemporaryDirectory(prefix="ugkp-sst-") as temporary:
        executable = Path(temporary) / "ugkp_sst_algebra_test"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-I",
                str(ROOT.parents[1] / "common"),
                str(SOURCE),
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(executable)], check=True)
