"""Ordered, rebuild-local wall traversal and exact all-face reference distances."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

APP = (SOURCE_ROOT / 'applications/CHMT')


def test_wall_faces_are_selected_once_per_rebuild():
    source = (APP / "mesh/Geometry.C").read_text()
    distance = source.split("m.wallDistance.assign(nc, 0);", 1)[1].split(
        "std::uint64_t h =", 1
    )[0]
    # Structural performance contract: this is intentionally separate from the
    # numerical equivalence test, which the former all-face scan also satisfies.
    before_cells, cells = distance.split("const auto cellWallDistance=", 1)
    assert "std::vector<int> wallFaces;" in before_cells
    assert "for (int f = 0; f<nf; ++f)" in before_cells
    assert "wallFaces.push_back(f)" in before_cells
    assert "for (int f:wallFaces)" in cells
    assert "boundaryKind" not in cells
    assert "static std::" not in distance


def test_all_wall_distances_match_ordered_all_face_reference(tmp_path):
    binary = tmp_path / "wall_faces"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(APP), "-I" + str(SOURCE_ROOT / "common"), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),
         str((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_wall_face_list.cpp')),
         "-o", str(binary)], check=True
    )
    result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    assert "wall-distance fixtures checked: 12" in result.stdout
