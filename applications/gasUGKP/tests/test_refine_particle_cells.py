from __future__ import annotations

import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ugkp_particle_restart import collect_restart, iter_restart_chunks              
from refine_particle_cells import refine_restart              


FLOAT_FIELDS = ("px", "py", "pz", "pux", "puy", "puz", "pT", "pTheta", "pd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_v5(path: Path) -> dict[str, np.ndarray]:
    data = {
        name: np.asarray([column + 0.25, column + 1.25, column + 2.25], dtype="<f8")
        for column, name in enumerate(FLOAT_FIELDS)
    }
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
            data["cell"][start:start + count].tofile(stream)
            data["status"][start:start + count].tofile(stream)
            data["rng"][start:start + count].tofile(stream)
            data["orig_id"][start:start + count].tofile(stream)
    return data


def test_local_refinement_is_deterministic_unique_and_conservative(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    output = tmp_path / "refined.dat"
    manifest_path = tmp_path / "manifest.json"
    source_data = write_v5(source)
    source_hash = sha256(source)

    manifest = refine_restart(
        source=source,
        output=output,
        cells={1},
        factor=4,
        legacy_parcel_mass=2.0,
        seed=12345,
        chunk_particles=3,
        manifest_path=manifest_path,
    )

    assert sha256(source) == source_hash
    assert manifest["conservation_passed"] is True
    assert manifest["source_particles"] == 3
    assert manifest["output_particles"] == 9
    assert manifest["selected_cells"] == [1]
    assert manifest["selected_source_particles"] == 2
    assert manifest_path.is_file()

    refined = collect_restart(output)
    assert len(refined["cell"]) == 9
    assert len(np.unique(refined["orig_id"])) == 9
    assert len(np.unique(refined["rng"])) == 9
    assert set(source_data["orig_id"]).issubset(set(refined["orig_id"]))

    for parent in range(3):
        original_id = source_data["orig_id"][parent]
        if source_data["cell"][parent] == 1:
            indices = np.flatnonzero(
                np.all(
                    np.column_stack([refined[name] for name in FLOAT_FIELDS])
                    == np.asarray([source_data[name][parent] for name in FLOAT_FIELDS]),
                    axis=1,
                )
                & (refined["cell"] == 1)
            )
            assert len(indices) == 4
            np.testing.assert_array_equal(refined["pm"][indices], np.full(4, 0.5))
            assert original_id in refined["orig_id"][indices]
        else:
            index = np.flatnonzero(refined["orig_id"] == original_id)
            assert len(index) == 1
            i = int(index[0])
            assert refined["cell"][i] == 2
            assert refined["pm"][i] == 2.0
            for name in FLOAT_FIELDS:
                assert refined[name][i].tobytes() == source_data[name][parent].tobytes()


def test_refinement_rejects_non_power_of_two_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    write_v5(source)
    with pytest.raises(ValueError, match="power of two"):
        refine_restart(source, tmp_path / "bad.dat", {1}, 3, 2.0, 7)

    output = tmp_path / "exists.dat"
    output.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        refine_restart(source, output, {1}, 2, 2.0, 7)
    assert output.read_bytes() == b"do not overwrite"


def test_reader_requires_explicit_legacy_mass_but_ugkp_embeds_pm(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    output = tmp_path / "weighted.dat"
    write_v5(source)
    with pytest.raises(ValueError, match="legacy parcel mass"):
        list(iter_restart_chunks(source))

    refine_restart(source, output, {1}, 2, 2.0, 9, chunk_particles=2)
    chunks = list(iter_restart_chunks(output))
    assert sum(len(chunk["pm"]) for chunk in chunks) == 5


def test_capacity_update_breaks_hardlink_without_mutating_alias(tmp_path: Path) -> None:
    source = tmp_path / "ugkwpProperties.source"
    case_copy = tmp_path / "ugkwpProperties"
    source.write_text("gpuResidentParticleCapacity 10;\nparcelMass 2;\n")
    os.link(source, case_copy)
    restart = tmp_path / "source.dat"
    write_v5(restart)

    refine_restart(
        restart,
        tmp_path / "weighted.dat",
        {1},
        2,
        2.0,
        12,
        properties_path=case_copy,
        capacity_reserve_fraction=0.05,
        capacity_reserve_minimum=100,
    )

    assert "gpuResidentParticleCapacity 10;" in source.read_text()
    assert "gpuResidentParticleCapacity 105;" in case_copy.read_text()
    assert source.stat().st_ino != case_copy.stat().st_ino
