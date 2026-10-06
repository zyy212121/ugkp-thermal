"""CUDA fixture generation stays testable on hosts without an NVIDIA toolkit."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("app,bits", [("gasUGKP", 64), ("FSH", 64), ("CHT", 64), ("CHT", 32)])
def test_corrected_cold_wall_fixture_preserves_other_frozen_oracles(tmp_path, app, bits):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/fixtures/operator_unification/differential.py"),
         str(ROOT), str(tmp_path), app, str(bits), "--generate-only"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    source = (tmp_path / "differential.cu").read_text()
    frozen = json.loads((ROOT / "tests/fixtures/operator_unification/baseline.json").read_text())
    names = ["solidEpsFromMomentDevice", "solidPressureFromMomentsDevice",
             "granularCollisionTauFromCellDevice", "riemannFacePrimitiveForGradient"]
    names += ["preparePressureProjectionCell" if app == "gasUGKP" else
              "atomicAddParticleWallEnergyByFace"]
    for name in names:
        assert frozen[app + ":" + name].replace(name, "baseline_" + name) in source
    assert frozen["drag"].replace("UGKWP_GPU_DRAG_ALGEBRA_CUH", "BASELINE_GPU_DRAG_ALGEBRA_CUH").replace("ugkwpGpuDragAlgebra", "baselineDrag") in source
    assert "baselineColdWall" not in source
    if app != "gasUGKP":
        for key, old, new in [
            ("thermalPressureCell", "applyCollisionalPressureProjectionCellAtomicKernel", "baselinePressureCell"),
            ("thermalPressureParticles", "applyCollisionalPressureProjectionParticlesAtomicKernel", "baselinePressureParticles"),
        ]:
            assert frozen[key].replace(old, new) in source
        # These are generator contracts only; actual numerical assertions run in
        # the CUDA fixture, and host scalar physics has separate executable tests.
        assert "coldScalar<<<1,1>>>" in source
        assert "Foam::gpuThermal::advanceColdWallProfile(" in source
        assert "seedColdReference(s,reference);" in source
        assert "uniform gas-only analytic exponential" in source
        assert "particle enthalpy + wall heat - gas input conservation" in source
        assert "rejected scalar ring mass unchanged" in source
        assert ("advanceColdWall1DThermalGroup<true>" in source) == (bits == 32)
    else:
        assert "coldScalar" not in source
    for placeholder in ["COLD_CALLS", "COLD_TEST", "WALL_TEST", "PRESSURE_TEST", "EXTRA_KERNELS", "STABLE_MODES"]:
        assert placeholder not in source
