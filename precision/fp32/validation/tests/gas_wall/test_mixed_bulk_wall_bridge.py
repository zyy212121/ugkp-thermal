"""Scheduled-checkpoint coverage of float bulk and the double coherent wall."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_mixed_bulk_keeps_double_coherent_wall(tmp_path):
    source = SOURCE_ROOT / "validation/tests/gas_wall/mixed_bulk_wall_bridge.cpp"
    executable = tmp_path / "mixed_bulk_wall_bridge"
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-I" + str(SOURCE_ROOT),
         "-I" + str(SOURCE_ROOT / "common"), str(source), "-o", str(executable)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)
