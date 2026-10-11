"""Host output-contract tests, not native execution evidence."""
from pathlib import Path
import subprocess

APP = Path(__file__).resolve().parents[1]

def test_accepted_output_contract(tmp_path):
    binary = tmp_path / "output-contract"
    build = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(APP), str(APP / "tests/test_coupled_output.cpp"), "-o", str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
