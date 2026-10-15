"""CPU execution of production cold-wall preparation, dispatch and transitions.

Eight synchronized std::thread lanes emulate CUDA shuffle/ballot collectives.
This is not native CUDA coverage: device codegen, FMA and launch/performance are
not tested. All thermal/preparation/selection/transition bodies are production.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

import pytest

ROOT = SOURCE_ROOT


@pytest.fixture(scope="module", params=[False, True],
                ids=["cht_fp64", "fsh_fp64"])
def contact_routes_executable(request, tmp_path_factory):
    fsh = request.param
    build = tmp_path_factory.mktemp(f"contact_routes_{fsh}")
    conductivity_source = (ROOT / "common/operators/updateLegacyGasBoundaryMirrorKernel.cuh").read_text()
    start = conductivity_source.index("template<class GasState>\n__device__ GPU_OPERATOR_REAL molecularGasConductivity")
    end = conductivity_source.index("\n}", start) + 2
    (build / "route_conductivity.cuh").write_text('#include "gasTransport/GasStateView.H"\n' + conductivity_source[start:end])
    executable = build / "contact_routes"
    result = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-pthread", "-Wall", "-Wextra",
         f"-DROUTE_FSH={int(fsh)}", "-I", str(build),
         "-I", str(ROOT / "common"), "-I", str(ROOT / "common/wall"),
         str((VALIDATION_ROOT / 'tests/fixtures/cold_wall_contact_routes.cpp')),
         "-o", str(executable)], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "deposit_area_conversion", "contact_arithmetic", "large_finalizer_energy", "zero_midpoint_area", "short_contact_interval", "zero_active_time",
    "deposited_positive_area", "deposited_zero_area_invalid", "gas_off",
    "detach_then_mobile", "collision_released_before_thermal",
    "completed_frozen_heating", "completed_frozen_cooling", "gas_off_profile_preserved",
    "gas_only_invalid_profile",
    "invalid_duration_zero", "invalid_duration_nan", "invalid_duration_inf",
    "invalid_peak_zero", "invalid_peak_one", "invalid_peak_nan",
    "invalid_damage_negative", "invalid_damage_nan", "invalid_damage_inf",
])
def test_production_cold_wall_contact_route(contact_routes_executable, case):
    result = subprocess.run([str(contact_routes_executable), case],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
