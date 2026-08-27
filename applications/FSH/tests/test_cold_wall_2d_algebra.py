import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cold_wall_2d_geometry_and_phase_algebra():
    source = ROOT / "tests" / "cold_wall_2d_algebra_test.cpp"
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
