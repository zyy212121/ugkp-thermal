"""CPU execution of production cold-wall preparation, dispatch and transitions.

Eight synchronized std::thread lanes emulate CUDA shuffle/ballot collectives.
This is not native CUDA coverage: device codegen, FMA and launch/performance are
not tested. All thermal/preparation/selection/transition bodies are production.
"""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[(32, False, False), (32, True, False), (64, False, False), (64, False, True)],
                ids=["cht_fp32", "cht_fp32_stable", "cht_fp64", "fsh_fp64"])
def contact_routes_executable(request, tmp_path_factory):
    bits, stable, fsh = request.param
    build = tmp_path_factory.mktemp(f"contact_routes_{bits}_{stable}")
    # Force the existing FP32 stable specialization without duplicating the
    # production kernel or its input preparation. The fallback stays intact.
    if stable:
        source = (ROOT / "common/wall/GpuColdWall1DDevice.cuh").read_text()
        needle = "bool valid = advanceColdWall1DThermalGroup\n"
        assert source.count(needle) == 2
        source = source.replace(needle, "bool valid = advanceColdWall1DThermalGroup<true>\n", 1)
        (build / "GpuColdWall1DDevice.cuh").write_text(source)
    conductivity_source = (ROOT / "common/operators/updateLegacyGasBoundaryMirrorKernel.cuh").read_text()
    start = conductivity_source.index("template<class GasState>\n__device__ GPU_OPERATOR_REAL molecularGasConductivity")
    end = conductivity_source.index("\n}", start) + 2
    (build / "route_conductivity.cuh").write_text('#include "gasTransport/GasStateView.H"\n' + conductivity_source[start:end])
    executable = build / "contact_routes"
    result = subprocess.run(
        ["g++", "-std=c++20", "-O2", "-pthread", "-Wall", "-Wextra",
         f"-DUGKWP_GPU_REAL_BITS={bits}", f"-DROUTE_FSH={int(fsh)}", "-I", str(build),
         "-I", str(ROOT / "common"), "-I", str(ROOT / "common/wall"),
         str(ROOT / "tests/fixtures/cold_wall_contact_routes.cpp"),
         "-o", str(executable)], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "zero_midpoint_area", "short_contact_interval", "zero_active_time",
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
