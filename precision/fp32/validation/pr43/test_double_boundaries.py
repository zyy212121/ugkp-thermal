"""Scheduled 2A/2B checks of wall-source and geometry precision boundaries."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT


@pytest.mark.parametrize("fixture", ["double_geometry_intersection", "double_wall_source"])
def test_double_precision_boundary(tmp_path, fixture):
    source = SOURCE_ROOT / "validation/pr43" / (fixture + ".cpp")
    executable = tmp_path / fixture
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I" + str(SOURCE_ROOT),
         "-I" + str(SOURCE_ROOT / "common"),
         "-I" + str(SOURCE_ROOT / "common/gasNumerics"),
         str(source), "-o", str(executable)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)
