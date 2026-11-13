from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def test_mixed_bulk_double_energy_eos_and_rk(tmp_path):
    binary = tmp_path / "mixed_eos_rk"
    source = ROOT / "validation/pr43/mixed_eos_rk.cpp"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I"+str(ROOT),
                    "-I"+str(ROOT/"common"), "-I"+str(ROOT/"common/gasNumerics"),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
