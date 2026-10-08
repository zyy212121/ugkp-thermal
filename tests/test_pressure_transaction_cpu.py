"""Execute production pressure transaction arithmetic on a single host thread.

CUDA qualifiers, indices and atomics are replaced only by serial equivalents.
The production preflight is copied verbatim up to its device-trap/launch adapter;
no equations or branches are replaced. This does not test CUDA scheduling,
parallel reductions, traps or launch ordering.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/pressure/transaction_cpu.cpp"
MODES = [
    "mobile_sorted", "mobile_compact", "mobile_split", "mobile_unsorted", "mobile_volume",
    "mixed_sorted", "mixed_compact", "mixed_split", "mixed_unsorted",
    "initial_mismatch", "nonfinite_initial", "nonfinite_particle", "negative_particle",
    "invalid_initial", "particle_overflow", "zero_initial_mismatch", "inactive", "empty", "global_failure",
    "actual_floor", "actual_velocity", "actual_particle_velocity", "invalid_count_negative", "invalid_count_precision", "invalid_count_range", "derived_overflow", "tiny_relative_tolerance", "canonical", "zero_identity",
    "unsorted_recovery", "invalid_status", "invalid_directory", "limiter",
]


@pytest.fixture(scope="module", params=["float", "double"])
def transaction_executable(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to execute the production pressure functions")
    directory = tmp_path_factory.mktemp("pressure_transaction_" + request.param)
    preflight = (ROOT / "common/GpuPressurePreflight.cuh").read_text()
    boundary = "__global__ void enforcePressurePreflightKernel"
    assert preflight.count(boundary) == 1, "preflight's host/device boundary changed"
    (directory / "PressurePreflightHost.cuh").write_text(preflight.split(boundary)[0])
    binary = directory / "check"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-DTEST_REAL=" + request.param, "-I", str(ROOT / "common"),
         "-I", str(directory), str(FIXTURE), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    return binary


@pytest.mark.parametrize("mode", MODES)
def test_production_pressure_transaction(transaction_executable, mode):
    run = subprocess.run([str(transaction_executable), mode], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
