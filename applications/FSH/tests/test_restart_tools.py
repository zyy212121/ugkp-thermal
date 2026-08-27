from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fsh_particle_restart import (              
    ALL_FIELDS,
    collect_restart,
    write_fsh_restart,
)
from refine_fsh_particle_cells import refine_restart              


def sample_chunk() -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for column, name in enumerate(
        ("px", "py", "pz", "pux", "puy", "puz", "pT", "pTheta", "pd")
    ):
        values[name] = np.asarray(
            [column + 0.25, column + 1.25, column + 2.25], dtype="<f8"
        )
    values["pTheta"] = np.asarray([0.25, 0.5, 0.75], dtype="<f8")
    values["pd"] = np.asarray([1.0e-4, 1.2e-4, 1.4e-4], dtype="<f8")
    values["pm"] = np.asarray([2.0, 3.0, 5.0], dtype="<f8")
    values["cell"] = np.asarray([1, 2, 1], dtype="<i4")
    values["status"] = np.ones(3, dtype="<i4")
    values["rng"] = np.asarray([101, 202, 303], dtype="<u8")
    values["orig_id"] = np.asarray([11, 22, 33], dtype="<u8")
    values["pStuck"] = np.asarray([0, 1, 1], dtype="<u1")
    values["pStuckFaceId"] = np.asarray([-1, 7555, -2], dtype="<i4")
    values["pDepositionArea"] = np.asarray([0.0, 3.5e-8, 0.0], dtype="<f4")
    assert set(values) == set(ALL_FIELDS)
    return values


def test_fsh_binary_round_trip_keeps_every_extended_field(tmp_path: Path) -> None:
    restart = tmp_path / "particles.dat"
    source = sample_chunk()
    write_fsh_restart(restart, [source], len(source["pm"]), chunk_particles=2)

    assert restart.read_bytes().startswith(b"UGKP_FSH_PARTICLES_SCHEMA1_BIN 3 2\n")
    restored = collect_restart(restart)
    for name in ALL_FIELDS:
        np.testing.assert_array_equal(restored[name], source[name])


def test_stopped_state_refinement_is_mass_and_contact_area_conservative(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.dat"
    output_path = tmp_path / "refined.dat"
    source = sample_chunk()
    write_fsh_restart(source_path, [source], 3, chunk_particles=2)

    manifest = refine_restart(
        source_path,
        output_path,
        cells={1},
        factor=4,
        legacy_parcel_mass=None,
        seed=3091,
        chunk_particles=3,
    )
    refined = collect_restart(output_path)

    assert manifest["format"] == "UGKP_FSH_PARTICLES_SCHEMA1_BIN"
    assert manifest["conservation_passed"] is True
    assert manifest["output_particles"] == 9
    assert len(np.unique(refined["orig_id"])) == 9
    assert len(np.unique(refined["rng"])) == 9

    for source_index in (0, 2):
        matches = (
            (refined["cell"] == source["cell"][source_index])
            & (refined["px"] == source["px"][source_index])
            & (refined["pStuck"] == source["pStuck"][source_index])
            & (refined["pStuckFaceId"] == source["pStuckFaceId"][source_index])
            & (refined["pDepositionArea"] == source["pDepositionArea"][source_index])
        )
        indices = np.flatnonzero(matches)
        assert len(indices) == 4
        np.testing.assert_array_equal(
            refined["pm"][indices],
            np.full(4, source["pm"][source_index] / 4.0),
        )

