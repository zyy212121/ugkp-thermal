"""Actual shared device headers on a serial host emulator, not a CUDA runtime test."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("adapter", ["gas_view", "legacy", "legacy_optional_species"])
def test_gas_only_view_runs_every_legacy_operator_without_application_state(tmp_path, bits, adapter):
    assert (ROOT / "common/gasTransport/GasStateView.H").is_file(), "missing independent gas-only state view"
    exe = tmp_path / f"gas{bits}"
    flags = ["-DLEGACY_REFERENCE"] if adapter.startswith("legacy") else []
    if adapter == "legacy_optional_species": flags.append("-DLEGACY_OPTIONAL_SPECIES")
    build = subprocess.run(["g++", *flags, "-std=c++17", "-O0", f"-DUGKWP_GPU_REAL_BITS={bits}", "-I"+str(ROOT / "common"), "-I"+str(ROOT / "common/gpu"), "-I"+str(ROOT / "common/gasNumerics"), str(HERE / "gas_state_probe.cpp"), "-o", str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == (HERE / f"legacy_gas_operator_fp{bits}.txt").read_text()
