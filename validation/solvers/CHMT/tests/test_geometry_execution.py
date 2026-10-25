"""Byte-exact stage/rebuild regression against the PR25 serial geometry oracle."""
import os
import subprocess
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT

APP = SOURCE_ROOT / "applications/CHMT"
TESTS = VALIDATION_ROOT / "solvers/CHMT/tests"


def test_geometry_execution_matches_serial_reference(tmp_path):
    for parallel in (False, True):
        binary = tmp_path / ("geometry-omp" if parallel else "geometry-serial")
        command = ["g++", "-std=c++17", "-O2", "-frounding-math", "-Wall", "-Wextra", "-Werror", "-pedantic",
                   "-I" + str(APP), "-I" + str(SOURCE_ROOT / "common"), "-I" + str(VALIDATION_ROOT / "solvers/CHMT"),
                   str(TESTS / "test_geometry_execution.cpp"), "-o", str(binary)]
        if parallel:
            command.append("-fopenmp")
        subprocess.run(command, check=True)
        for threads in ((1, 4) if parallel else (1,)):
            result = subprocess.run([str(binary)], env={**os.environ, "OMP_NUM_THREADS": str(threads)},
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            assert "exact geometry reference comparisons: 128" in result.stdout
