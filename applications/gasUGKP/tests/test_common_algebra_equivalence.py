from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("common_algebra_equivalence_test.cpp")


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
            str(SOURCE),
            "-o",
            str(executable),
        ),
        check=True,
    )
    subprocess.run((str(executable),), check=True)
