"""Standalone CPU surface algebra checks, not the required OpenFOAM validation."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dry_material_surface(tmp_path):
    binary = tmp_path / "surface"
    subprocess.run([
        "g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
        "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
        "-I" + str(ROOT / "applications/CHMT"),
        str(ROOT / "validation/tests/fixtures/dry_material/surface.cpp"),
        "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
