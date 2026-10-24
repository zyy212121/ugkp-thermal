"""Production CPU adapters, independent of the GPU runtime."""
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_dry_material_adapter(tmp_path):
    binary = tmp_path / "adapter"
    subprocess.run([
        "g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-Wno-misleading-indentation", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
        "-I" + str(ROOT / "applications/CHMT"), "-I" + str(ROOT / "common"),
        str(ROOT / "validation/tests/fixtures/dry_material/adapter.cpp"),
        str(ROOT / "applications/CHMT/materials/MaterialTransport.C"),
        "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)


def test_dry_material_adapter_three_species(tmp_path):
    binary = tmp_path / "adapter_ns3"
    subprocess.run([
        "g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-Wno-misleading-indentation", "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
        "-DUGKWP_GAS_SPECIES=3",
        "-I" + str(ROOT / "applications/CHMT"), "-I" + str(ROOT / "common"),
        str(ROOT / "validation/tests/fixtures/dry_material/adapter_ns3.cpp"),
        str(ROOT / "applications/CHMT/materials/MaterialTransport.C"),
        "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
