"""Host execution of the real shared enthalpy helper, in both precisions."""
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[32, 64])
def cold_wall_executable(request, tmp_path_factory):
    executable = tmp_path_factory.mktemp(f"cold_wall_{request.param}") / "energy_test"
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
         f"-DUGKWP_GPU_REAL_BITS={request.param}", "-I", str(ROOT / "common"),
         "-I", str(ROOT / "common/wall"),
         str(ROOT / "tests/fixtures/cold_wall_1d_energy_test.cpp"), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("mode", ["constant_cp", "nonlinear", "rollback", "gas_uniform",
                                  "gas_phase", "gas_zero", "gas_rollback", "source_limits",
                                  "phase_refinement", "separate_gas_duration"])
def test_cold_wall_enthalpy_equation(cold_wall_executable, mode):
    result = subprocess.run([str(cold_wall_executable), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bits", [32, 64])
def test_shared_cuda_wrapper_cpp_syntax(tmp_path, bits):
    """Compile the actual wrapper body with declared intrinsics; no GPU emulation."""
    source = tmp_path / "wrapper.cpp"
    source.write_text(r'''
#define __device__
#include "GpuColdWallSolidification.H"
template<class T> T __shfl_sync(unsigned, T, int, int);
template<class T> T __shfl_up_sync(unsigned, T, int, int);
template<class T> T __shfl_down_sync(unsigned, T, int, int);
int __all_sync(unsigned, int);
unsigned __ballot_sync(unsigned, int);
int __ffs(unsigned);
struct DeviceState {
    int coldWallSolidificationEnabled;
    float *pColdNodeSpecificEnthalpy, *pColdRingSolidMass, *pColdFrozenArea;
    GpuTime* pColdContactAge;
    Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters;
};
#define GPU_COLD_WALL_1D_ALGEBRA_ONLY
#include "GpuColdWall1DDevice.cuh"
void instantiate(DeviceState& state) {
    GpuReal temperature = 0, area = 0, energy = 0;
    advanceColdWall1DThermalGroup(state, 0, 0, 255, GPU_R(1e-12), GPU_R(3e-9),
        GPU_R(2e-8), GPU_R(1e-8), 1e-5, GPU_R(.4), 1e-7, GPU_R(300),
        GPU_R(13000), GPU_R(.1), GPU_R(1500), GPU_R(3e-4), temperature, area, energy);
#if UGKWP_GPU_REAL_BITS == 32
    advanceColdWall1DThermalGroup<true>(state, 0, 0, 255, GPU_R(1e-12), GPU_R(3e-9),
        GPU_R(2e-8), GPU_R(1e-8), 1e-5, GPU_R(.4), 1e-7, GPU_R(300),
        GPU_R(13000), GPU_R(.1), GPU_R(1500), GPU_R(3e-4), temperature, area, energy);
#endif
}
''')
    result = subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
         f"-DUGKWP_GPU_REAL_BITS={bits}", "-I", str(ROOT / "common"),
         "-I", str(ROOT / "common/wall"), str(source)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_assembly_uses_one_bulk_source_and_old_enthalpy():
    body = (ROOT / "common/wall/GpuColdWall1DAdvance.inl").read_text()
    loop = body.index("iteration < s.coldWallSolidificationParameters.nonlinearIterations")
    assert body.index("coldWallGasSpecificEnthalpyIncrement") < loop
    assert body.count("coldWallGasSpecificEnthalpyIncrement") == 1
    assert "oldEnthalpy - candidateEnthalpy" in body
    assert "diagonal += gasConductanceWK" not in body
    assert "gasConductanceWK*gasTemperatureK" not in body
    assert "lane == 7 ? gasConductanceWK" not in body
    assert "candidateEnthalpy = oldEnthalpy\n          + gasSpecificEnthalpyIncrement" in body
    inputs = (ROOT / "common/wall/GpuColdWall1DInputs.inl").read_text()
    assert "lane == 0\n         && s.solveParticleTemperature != 0" in inputs
    assert "s.particleGasHeatTransferModelId != 0" in inputs
    assert "gasConductanceWK = __shfl_sync(mask, gasConductanceWK, 0, 8)" in inputs
    wrappers = (ROOT / "common/wall/GpuColdWall1DDevice.cuh").read_text()
    assert wrappers.count('#include "GpuColdWall1DAdvance.inl"') == 2
    for application, backend in [("FSH", "private_backend"), ("CHT", "gpu")]:
        caller = (ROOT / "applications" / application / backend / "GpuResidentStrict.cu").read_text()
        assert '#include "../../../common/wall/GpuColdWall1DDevice.cuh"' in caller
