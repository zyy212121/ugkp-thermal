"""Real CPU geometry: positive precision margins, unlimited workspace, Empty representation."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT

@pytest.fixture(scope="module")
def geometry_probe(tmp_path_factory):
    executable = tmp_path_factory.mktemp("geometry-policy") / "probe"
    app = SOURCE_ROOT / "applications/CHMT"
    command = ["g++", "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
               "-Wno-misleading-indentation", "-Wno-mismatched-new-delete",
               "-I" + str(app), "-I" + str(SOURCE_ROOT / "common"),
               str(VALIDATION_ROOT / "tests/fixtures/geometry_policy/probe.cpp"),
               str(app / "mesh/Geometry.C"), "-o", str(executable)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable

@pytest.mark.parametrize("case", ["margin", "tiny-margin", "rotation", "caps", "empty", "empty-nonfinite", "multiple-cover", "parity"])
def test_geometry_policy(geometry_probe, case):
    result = subprocess.run([str(geometry_probe), case], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_geometry_matches_previous_uncapped_solve(geometry_probe, tmp_path):
    """Historical cached and current streamed modes retain exact physical output."""
    revision = "ce55d18995a60de932ea63ba4b954afc68629139"
    app = SOURCE_ROOT / "applications/CHMT"
    (tmp_path / "mesh").mkdir()
    for name in ("Geometry.C", "SweepWorkspace.H", "SweepConstraints.H", "EmptyExtrusion.H"):
        content = subprocess.check_output(["git", "show", revision + ":applications/CHMT/mesh/" + name], cwd=SOURCE_ROOT)
        (tmp_path / "mesh" / name).write_bytes(content)
    baseline = tmp_path / "baseline"
    command = ["g++", "-std=c++14", "-O2", "-I" + str(tmp_path), "-I" + str(app),
               "-I" + str(SOURCE_ROOT / "common"),
               str(VALIDATION_ROOT / "tests/fixtures/geometry_policy/probe.cpp"),
               str(tmp_path / "mesh/Geometry.C"), "-o", str(baseline)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    before = subprocess.run([str(baseline), "parity"], check=True, capture_output=True, text=True)
    after = subprocess.run([str(geometry_probe), "parity"], check=True, capture_output=True, text=True)
    assert before.stdout == after.stdout
    for case in ("margin", "tiny-margin", "rotation", "caps", "empty-nonfinite"):
        old = subprocess.run([str(baseline), case], capture_output=True, text=True)
        assert old.returncode != 0, "historical control must expose " + case
        assert "FAIL:" in old.stderr


def test_triangle_distance_uses_actual_geometry(tmp_path):
    app = SOURCE_ROOT / "applications/CHMT"
    source = VALIDATION_ROOT / "tests/fixtures/geometry_policy/triangle_distance.cpp"
    executable = tmp_path / "triangle-distance"
    built = subprocess.run(["g++", "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                            "-I" + str(app), "-I" + str(SOURCE_ROOT / "common"), str(source),
                            "-o", str(executable)], text=True, capture_output=True)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(executable)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
