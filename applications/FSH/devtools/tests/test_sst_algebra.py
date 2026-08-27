from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("ugkp_sst_algebra_test.cpp")


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
                str(SOURCE),
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(executable)], check=True)
