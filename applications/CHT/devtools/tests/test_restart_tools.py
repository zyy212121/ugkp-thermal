from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from ugkp_particle_restart import collect_restart, iter_restart_chunks              
from refine_particle_cells import refine_restart              


FLOAT_FIELDS = ("px", "py", "pz", "pux", "puy", "puz", "pT", "pTheta", "pd")


def write_v5(path: Path) -> dict[str, np.ndarray]:
    data = {
        name: np.asarray([column + 0.25, column + 1.25, column + 2.25], dtype="<f8")
        for column, name in enumerate(FLOAT_FIELDS)
    }
    data["pTheta"] = np.asarray([0.25, 0.5, 0.75], dtype="<f8")
    data["pd"] = np.asarray([1.0e-4, 1.2e-4, 1.4e-4], dtype="<f8")
    data["cell"] = np.asarray([1, 2, 1], dtype="<i4")
    data["status"] = np.ones(3, dtype="<i4")
    data["rng"] = np.asarray([101, 202, 303], dtype="<u8")
    data["orig_id"] = np.asarray([11, 22, 33], dtype="<u8")
    with path.open("wb") as stream:
        stream.write(b"UGKP_PARTICLES_SCHEMA5_BIN 3 2\n")
        for start, count in ((0, 2), (2, 1)):
            stream.write(struct.pack("<I", count))
            for name in FLOAT_FIELDS:
                data[name][start:start + count].tofile(stream)
            for name in ("cell", "status", "rng", "orig_id"):
                data[name][start:start + count].tofile(stream)
    return data


def test_weighted_round_trip_and_cell_refinement_are_conservative(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    output = tmp_path / "weighted.dat"
    original = write_v5(source)
    manifest = refine_restart(source, output, {1}, 4, 2.0, 3001, chunk_particles=3)
    restored = collect_restart(output)

    assert manifest["format"] == "UGKP_PARTICLES_SCHEMA1_BIN"
    assert manifest["conservation_passed"] is True
    assert manifest["output_particles"] == 9
    assert len(np.unique(restored["orig_id"])) == 9
    assert len(np.unique(restored["rng"])) == 9
    for parent in (0, 2):
        match = restored["cell"] == original["cell"][parent]
        for name in FLOAT_FIELDS:
            match &= restored[name] == original[name][parent]
        indices = np.flatnonzero(match)
        assert len(indices) == 4
        np.testing.assert_array_equal(restored["pm"][indices], np.full(4, 0.5))


def test_legacy_read_requires_explicit_mass_and_output_is_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    write_v5(source)
    with pytest.raises(ValueError, match="legacy parcel mass"):
        list(iter_restart_chunks(source))
    output = tmp_path / "exists.dat"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        refine_restart(source, output, {1}, 2, 2.0, 12)
    assert output.read_bytes() == b"preserve"

