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
from fsh_particle_restart_schema6 import (
    ALL_FIELDS as SCHEMA6_FIELDS,
    collect_restart as collect_schema6_restart,
    write_fsh_restart as write_schema6_restart,
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


def test_schema6_refinement_preserves_full_wall_state(tmp_path: Path) -> None:
    count = 3
    source = {
        name: np.asarray([0.1 + i, 0.2 + i, 0.3 + i], dtype="<f8")
        for i, name in enumerate(("px", "py", "pz", "pux", "puy", "puz", "pT"))
    }
    source["pTheta"] = np.asarray([0.25, 0.5, 0.75], dtype="<f8")
    source["pd"] = np.asarray([1e-4, 1.2e-4, 1.4e-4], dtype="<f8")
    source["pm"] = np.asarray([2.0, 3.0, 5.0], dtype="<f8")
    source["cell"] = np.asarray([1, 2, 1], dtype="<i4")
    source["status"] = np.ones(count, dtype="<i4")
    source["rng"] = np.asarray([101, 202, 303], dtype="<u8")
    source["orig_id"] = np.asarray([11, 22, 33], dtype="<u8")
    source["pStuck"] = np.asarray([0, 1, 3], dtype="<u1")
    source["pStuckFaceId"] = np.asarray([-1, 7555, 7556], dtype="<i4")
    source["pDepositionArea"] = np.asarray([0.0, 3.5e-8, 2e-8], dtype="<f4")
    source["pContactDuration"] = np.asarray([1.0 + 2.0**-40, 2.0, 3.0], dtype="<f8")
    source["pContactMaximumArea"] = np.asarray([0.0, 4e-8, 5e-8], dtype="<f4")
    source["pContactPeakFraction"] = np.asarray([0.0, 0.25, 0.4], dtype="<f4")
    source["pColdNodeSpecificEnthalpy"] = np.arange(count * 8, dtype="<f4").reshape(count, 8) + 1e6
    source["pColdRingSolidMass"] = np.arange(count * 8, dtype="<f4").reshape(count, 8) + 2e6
    source["pColdFrozenArea"] = np.asarray([0.0, 1e-8, 2e-8], dtype="<f4")
    source["pColdContactAge"] = np.asarray([1.0 + 2.0**-40, 2.0, 3.0], dtype="<f8")
    source["pCold2DNodeSpecificEnthalpy"] = np.arange(count * 64, dtype="<f4").reshape(count, 64) + 3e6
    source["pCold2DRingContactAge"] = np.full((count, 8), 1.0 + 2.0**-40, dtype="<f8")
    source["pCold2DFrozenArea"] = np.asarray([0.0, 3e-8, 4e-8], dtype="<f4")
    assert set(source) == set(SCHEMA6_FIELDS)

    source_path = tmp_path / "schema6_source.dat"
    output_path = tmp_path / "schema6_refined.dat"
    write_schema6_restart(source_path, [source], count, chunk_particles=2, schema=6)
    manifest = refine_restart(source_path, output_path, {1}, 4, None, 3001, chunk_particles=3)
    refined = collect_schema6_restart(output_path)

    assert output_path.read_bytes().startswith(b"UGKP_FSH_PARTICLES_SCHEMA6_BIN 9 3\n")
    assert manifest["source_format"] == "UGKP_FSH_PARTICLES_SCHEMA6_BIN"
    assert manifest["format"] == "UGKP_FSH_PARTICLES_SCHEMA6_BIN"
    assert manifest["conservation_passed"] is True
    assert refined["pContactDuration"].dtype == np.dtype("<f8")
    assert refined["pCold2DRingContactAge"].dtype == np.dtype("<f8")
    for source_index in (0, 2):
        matches = ((refined["cell"] == source["cell"][source_index])
                   & (refined["px"] == source["px"][source_index])
                   & (refined["pStuck"] == source["pStuck"][source_index]))
        indices = np.flatnonzero(matches)
        assert len(indices) == 4
        for name in SCHEMA6_FIELDS:
            if name not in ("pm", "orig_id", "rng"):
                expected = source[name][source_index]
                expected = np.broadcast_to(expected, refined[name][indices].shape)
                np.testing.assert_array_equal(refined[name][indices], expected)
        np.testing.assert_array_equal(refined["pm"][indices], np.full(4, source["pm"][source_index] / 4.0))
