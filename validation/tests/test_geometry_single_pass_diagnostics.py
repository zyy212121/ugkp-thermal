"""Fixed-interval swept Newton geometry and residual/physical diagnostic semantics."""
import subprocess

from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT


def test_geometry_single_pass_diagnostics(tmp_path):
    app = SOURCE_ROOT / "applications/CHMT"
    source = VALIDATION_ROOT / "tests/fixtures/geometry_single_pass/probe.cpp"
    executable = tmp_path / "geometry-single-pass"
    command = [
        "g++", "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
        "-Wno-misleading-indentation", "-Wno-mismatched-new-delete",
        "-I" + str(app), "-I" + str(SOURCE_ROOT / "common"),
        str(source), str(app / "mesh/Geometry.C"), "-o", str(executable),
    ]
    built = subprocess.run(command, text=True, capture_output=True)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(executable)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: fixed-interval geometry" in result.stdout


def test_geometry_has_no_terminal_completion_family():
    mesh = SOURCE_ROOT / "applications/CHMT/mesh"
    source = (mesh / "Geometry.C").read_text()
    for removed in ("terminalTail", "auto complete=", "<Terminal>"):
        assert removed not in source
    for file in mesh.glob("*.H"):
        assert "template<bool Terminal" not in file.read_text()
    for removed in ("completeGeometry", "completeStageGeometry", "completeFaceSweeps", "completeCoupledMeshes"):
        assert removed not in source
    assert "minimumNormCorrection" in source
    assert "edgeSweepQuadrature" in source
