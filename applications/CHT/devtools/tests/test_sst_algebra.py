from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("sst_algebra_test.cpp")


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
