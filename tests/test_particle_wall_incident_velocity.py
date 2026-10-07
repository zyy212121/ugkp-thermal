"""Host regressions for the production face walk and impact constitutive laws.

This covers the CHT FP32/FP64 and FSH FP64 adapters, not CUDA code generation,
GPU execution, or the physical validity of the existing moving-wall rebound law.
"""
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[(32, 0), (64, 0), (64, 1)],
                ids=["cht_fp32", "cht_fp64", "fsh_fp64"])
def incident_velocity_executable(request, tmp_path_factory):
    bits, fsh = request.param
    executable = tmp_path_factory.mktemp(f"incident_{bits}_{fsh}") / "incident"
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
         "-Werror", "-Wno-unused-parameter", "-pedantic", f"-DUGKWP_GPU_REAL_BITS={bits}",
         f"-DINCIDENT_FSH={fsh}", "-I", str(ROOT / "common"),
         "-I", str(ROOT / "common/wall"),
         str(ROOT / "tests/fixtures/particle_tracking/wall_incident_velocity.cpp"),
         "-o", str(executable)], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("scenario", [
    "constant", "force_zero_endpoint", "drag_zero_endpoint", "reversal",
    "oblique", "internal_hops", "periodic_hop", "prior_reflection", "moving_wall",
])
@pytest.mark.parametrize("wall_model", [1, 3, 4], ids=["rebound", "cold_1d", "cold_2d"])
@pytest.mark.parametrize("deposit", [0, 1], ids=["below_threshold", "above_threshold"])
def test_current_segment_drives_contact_and_saved_rebound(
        incident_velocity_executable, scenario, wall_model, deposit):
    result = subprocess.run(
        [str(incident_velocity_executable), scenario, str(wall_model), str(deposit)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("scenario", [
    "invalid_relative_speed", "invalid_diameter", "invalid_temperature",
    "invalid_candidate", "bound_deposited", "bound_rebound", "bound_deposit",
])
def test_contact_guards_and_existing_bound_states(incident_velocity_executable, scenario):
    result = subprocess.run([str(incident_velocity_executable), scenario],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
