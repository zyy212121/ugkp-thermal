"""Regression for independent Pr with fixed mixture wall composition."""
import shutil
import pytest
from pr_wall_regression import run

@pytest.mark.parametrize("native", [False, True], ids=["host", "cuda"])
def test_independent_pr_complete_wall_energy_flux(tmp_path, native):
    if native and not shutil.which("/usr/local/cuda/bin/nvcc"):
        pytest.skip("CUDA compiler unavailable")
    run(tmp_path / "probe", native=native)
