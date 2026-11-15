"""Narrow production CUDA pool tests, not solver-level thermal validation."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from validation.tools.paths import SOURCE_ROOT


def test_cold_wall_reorder_and_snapshot_routes():
    root = SOURCE_ROOT
    reorder = (root / "common/GpuCellLocalThermalFields.cuh").read_text()
    assert "s.compactPColdHandle[dst]=s.pColdHandle[i]" in reorder
    assert "swapParticlePointerDevice(s.pColdHandle,s.compactPColdHandle)" in reorder
    for prefix in ("", "precision/fp32/"):
        resident = (root / prefix / "common/thermal/GpuResidentStrict.cu").read_text()
        start = resident.index("UGKP_DEV_PROBE_ENTER(ProbeInjection)")
        stop = resident.index("injectBoundaryParticlesKernel", start)
        pre = resident[start:stop]
        for old_scan in ("clearColdWall1DUsedPrefixKernel", "markColdWall1DHandlesKernel",
                         "collectColdWall1DFreeKernel", "assignColdWall1DFreeKernel"):
            assert old_scan not in resident
        assert "gatherColdWall1DStateKernel" in resident
        assert "initialiseColdWall1DHandlesKernel" in resident
    commit = (root / "common/GpuParticleBufferCommit.cuh").read_text()
    assert "recordColdWall1DHighWater(s)" not in commit
    injection = (root / "common/operators/injectBoundaryParticlesKernel.cuh").read_text()
    assert "allocateColdWall1DHandle(s,slot)" in injection
    assert injection.index("slot >= s.particleCapacity") < injection.index("allocateColdWall1DHandle")
    pool = (root / "common/wall/GpuColdWall1DHandles.cuh").read_text()
    assert "atomicExch(s.pColdHandle+i,-1)" in pool
    assert "atomicSub(s.coldFreeCounts,1)-1" in pool
    adapter = (root / "common/thermal/ParticleStateAdapter.cuh").read_text()
    assert "coldWallFieldSlot(s,s.Field,i)" in adapter


def test_cold_wall_handle_cuda_lifecycle(tmp_path):
    # Explicit opt-in avoids taking the shared GPU during unrelated pytest runs.
    if os.environ.get("UGKP_RUN_COLD_HANDLE_CUDA") != "1":
        pytest.skip("set UGKP_RUN_COLD_HANDLE_CUDA=1 for the narrow GPU lifecycle fixture")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA toolkit unavailable")
    source = SOURCE_ROOT / "validation/tests/cold_wall_handle_lifecycle.cu"
    executable = tmp_path / "cold_wall_handle_lifecycle"
    built = subprocess.run(
        [nvcc, "-std=c++17", "-O2", "-I", str(SOURCE_ROOT / "common"),
         str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=180,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lifecycle CUDA fixture passed" in result.stdout
