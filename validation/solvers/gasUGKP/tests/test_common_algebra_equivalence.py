from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


from pathlib import Path
import subprocess


ROOT = SOURCE_ROOT
SOURCE = (VALIDATION_ROOT / 'solvers/gasUGKP/tests/common_algebra_equivalence_test.cpp')


def test_common_algebra_is_bitwise_equivalent_to_frozen_formulas(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "common_algebra_equivalence_test"
    subprocess.run(
        (
            "g++",
            "-std=c++17",
            "-O3",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "common"),
            str(SOURCE),
            "-o",
            str(executable),
        ),
        check=True,
    )
    subprocess.run((str(executable),), check=True)
