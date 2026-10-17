"""One requested interval and one SDIRK execution; no recovery algorithms."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_official_single_sdirk_result(tmp_path):
    executable = tmp_path / "single-sdirk"
    subprocess.run(
        ["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
         "-I", str(ROOT), str(ROOT / "validation/diagnostics/single_sdirk.cpp"),
         "-o", str(executable)], check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_removed_recovery_sources_are_absent():
    integrator = (ROOT / "common/chemistry/LocalStiffIntegrator.H").read_text()
    for retired in ("continueImplicitStage", "advanceClosedReactorTrial", "stepError",
                    "damping", "maximumLineSearchIterations", "maximumRejectedSteps"):
        assert retired not in integrator
    interval = (ROOT / "common/GpuGasAdvance.cuh").read_text()
    for retired in ("rollbackInterval", "continueTerminal", "substep", "maximumAttempts"):
        assert retired not in interval
    policy = (ROOT / "applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh").read_text()
    for retired in ("snapshotWallRecords", "rollbackInterval", "retryableFailure", "continueTerminal"):
        assert retired not in policy
