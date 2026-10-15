#!/usr/bin/env python3
                                                                           

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ugkp_particle_restart import (
    ALL_FIELDS,
    FLOAT_FIELDS,
    iter_restart_chunks,
    sha256,
    write_ugkp_restart,
)


INVARIANT_NAMES = ("mass", "mom_x", "mom_y", "mom_z", "energy", "mass_d", "mass_T")
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values = (values + np.uint64(0x9E3779B97F4A7C15)) & MASK64
        values = ((values ^ (values >> np.uint64(30)))
                  * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        values = ((values ^ (values >> np.uint64(27)))
                  * np.uint64(0x94D049BB133111EB)) & MASK64
    return values ^ (values >> np.uint64(31))


@dataclass
class SnapshotStats:
    particle_count: int = 0
    counts: dict[int, int] = field(default_factory=dict)
    moments: dict[int, np.ndarray] = field(default_factory=dict)
    scales: dict[int, np.ndarray] = field(default_factory=dict)
    global_moments: np.ndarray = field(
        default_factory=lambda: np.zeros(len(INVARIANT_NAMES), dtype=np.longdouble)
    )
    global_scales: np.ndarray = field(
        default_factory=lambda: np.zeros(len(INVARIANT_NAMES), dtype=np.longdouble)
    )

    def add(self, chunk: dict[str, np.ndarray], source: Path) -> None:
        count = len(chunk["pm"])
        if np.any(chunk["status"] != 1):
            raise ValueError(f"{source}: offline refinement requires compact active status=1")
        if np.any(chunk["cell"] < 0):
            raise ValueError(f"{source}: negative particle cell id")
        for name in FLOAT_FIELDS:
            if not np.all(np.isfinite(chunk[name])):
                raise ValueError(f"{source}: non-finite physical field {name}")
        if np.any(chunk["pTheta"] < 0) or np.any(chunk["pd"] <= 0):
            raise ValueError(f"{source}: invalid theta or diameter")

        m = chunk["pm"].astype(np.longdouble)
        ux = chunk["pux"].astype(np.longdouble)
        uy = chunk["puy"].astype(np.longdouble)
        uz = chunk["puz"].astype(np.longdouble)
        theta = chunk["pTheta"].astype(np.longdouble)
        values = np.column_stack(
            (
                m,
                m * ux,
                m * uy,
                m * uz,
                m * (np.longdouble(0.5) * (ux * ux + uy * uy + uz * uz)
                     + np.longdouble(1.5) * theta),
                m * chunk["pd"].astype(np.longdouble),
                m * chunk["pT"].astype(np.longdouble),
            )
        )
        scales = np.abs(values)
        self.global_moments += np.sum(values, axis=0, dtype=np.longdouble)
        self.global_scales += np.sum(scales, axis=0, dtype=np.longdouble)
        self.particle_count += count

        order = np.argsort(chunk["cell"], kind="stable")
        ordered_cells = chunk["cell"][order]
        starts = np.flatnonzero(
            np.r_[True, ordered_cells[1:] != ordered_cells[:-1]]
        )
        cells = ordered_cells[starts]
        cell_values = np.add.reduceat(values[order], starts, axis=0)
        cell_scales = np.add.reduceat(scales[order], starts, axis=0)
        cell_counts = np.diff(np.r_[starts, count])
        for index, cell_value in enumerate(cells):
            cell = int(cell_value)
            self.counts[cell] = self.counts.get(cell, 0) + int(cell_counts[index])
            if cell not in self.moments:
                self.moments[cell] = np.zeros(len(INVARIANT_NAMES), dtype=np.longdouble)
                self.scales[cell] = np.zeros(len(INVARIANT_NAMES), dtype=np.longdouble)
            self.moments[cell] += cell_values[index]
            self.scales[cell] += cell_scales[index]


def scan_restart(
    path: Path,
    legacy_parcel_mass: float | None,
) -> tuple[SnapshotStats, np.ndarray, np.ndarray]:
    stats = SnapshotStats()
    ids: list[np.ndarray] = []
    rng: list[np.ndarray] = []
    for chunk in iter_restart_chunks(path, legacy_parcel_mass):
        stats.add(chunk, path)
        ids.append(chunk["orig_id"].copy())
        rng.append(chunk["rng"].copy())
    all_ids = np.concatenate(ids) if ids else np.empty(0, dtype=np.uint64)
    all_rng = np.concatenate(rng) if rng else np.empty(0, dtype=np.uint64)
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError(f"{path}: duplicate input origId")
    if len(np.unique(all_rng)) != len(all_rng):
        raise ValueError(f"{path}: duplicate input RNG state")
    return stats, all_ids, all_rng


def _relative_residual(error: np.longdouble, scale: np.longdouble) -> float:
    floor = np.longdouble(np.finfo(np.float64).tiny)
    return float(abs(error) / max(abs(scale), floor))


def compare_stats(
    before: SnapshotStats,
    after: SnapshotStats,
    selected: set[int],
    factor: int,
    tolerance: float,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    maximum = 0.0
    if set(before.counts) != set(after.counts):
        raise ValueError("cell population support changed during refinement")
    for cell in sorted(before.counts):
        expected_count = before.counts[cell] * (factor if cell in selected else 1)
        if after.counts[cell] != expected_count:
            raise ValueError(
                f"cell {cell}: expected {expected_count} parcels, found {after.counts[cell]}"
            )
        for component, name in enumerate(INVARIANT_NAMES):
            residual = after.moments[cell][component] - before.moments[cell][component]
            scale = max(before.scales[cell][component], after.scales[cell][component])
            relative = _relative_residual(residual, scale)
            maximum = max(maximum, relative)
            rows.append(
                {
                    "cell": cell,
                    "invariant": name,
                    "before": f"{before.moments[cell][component]:.21e}",
                    "after": f"{after.moments[cell][component]:.21e}",
                    "residual": f"{residual:.21e}",
                    "scaled_residual": f"{relative:.17e}",
                }
            )
            if relative > tolerance:
                raise ValueError(
                    f"cell {cell} {name} conservation residual {relative:.3e} "
                    f"exceeds {tolerance:.3e}"
                )
    for component, name in enumerate(INVARIANT_NAMES):
        residual = after.global_moments[component] - before.global_moments[component]
        scale = max(before.global_scales[component], after.global_scales[component])
        relative = _relative_residual(residual, scale)
        maximum = max(maximum, relative)
        if relative > tolerance:
            raise ValueError(
                f"global {name} conservation residual {relative:.3e} exceeds {tolerance:.3e}"
            )
    return maximum, rows


def _refined_chunks(
    source: Path,
    legacy_parcel_mass: float | None,
    selected: set[int],
    factor: int,
    new_ids: np.ndarray,
    new_rng: np.ndarray,
) -> Iterator[dict[str, np.ndarray]]:
    cursor = 0
    exponent = factor.bit_length() - 1
    for chunk in iter_restart_chunks(source, legacy_parcel_mass):
        selected_parent = np.isin(chunk["cell"], np.asarray(sorted(selected), dtype=np.int32))
        repeats = np.where(selected_parent, factor, 1).astype(np.int64)
        base = np.repeat(np.arange(len(repeats), dtype=np.int64), repeats)
        group_start = np.repeat(np.cumsum(repeats) - repeats, repeats)
        clone_index = np.arange(len(base), dtype=np.int64) - group_start
        refined: dict[str, np.ndarray] = {
            name: np.ascontiguousarray(chunk[name][base]) for name in ALL_FIELDS
        }
        refined_selected = selected_parent[base]
        refined["pm"] = refined["pm"].copy()
        refined["pm"][refined_selected] = np.ldexp(
            refined["pm"][refined_selected], -exponent
        )
        extra = clone_index > 0
        extra_count = int(np.count_nonzero(extra))
        if extra_count:
            stop = cursor + extra_count
            refined["orig_id"] = refined["orig_id"].copy()
            refined["rng"] = refined["rng"].copy()
            refined["orig_id"][extra] = new_ids[cursor:stop]
            refined["rng"][extra] = new_rng[cursor:stop]
            cursor = stop
        yield refined
    if cursor != len(new_ids):
        raise RuntimeError(f"generated {cursor} clone records, expected {len(new_ids)}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def update_capacity(
    properties_path: Path,
    output_particles: int,
    reserve_fraction: float,
    reserve_minimum: int,
) -> int:
    if reserve_fraction < 0 or reserve_minimum < 0:
        raise ValueError("capacity reserve must be non-negative")
    reserve = max(int(math.ceil(output_particles * reserve_fraction)), reserve_minimum)
    capacity = output_particles + reserve
    source = properties_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^(\s*gpuResidentParticleCapacity\s+)(\d+)(\s*;)")
    updated, count = pattern.subn(rf"\g<1>{capacity}\g<3>", source, count=1)
    if count != 1:
        raise ValueError(f"{properties_path}: gpuResidentParticleCapacity entry not found")
    mode = properties_path.stat().st_mode
    temporary = properties_path.with_name(f".{properties_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, properties_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return capacity


def refine_restart(
    source: Path,
    output: Path,
    cells: set[int],
    factor: int,
    legacy_parcel_mass: float | None,
    seed: int,
    chunk_particles: int = 262144,
    manifest_path: Path | None = None,
    properties_path: Path | None = None,
    capacity_reserve_fraction: float = 0.05,
    capacity_reserve_minimum: int = 100000,
    tolerance: float = 5.0e-13,
) -> dict[str, object]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("source and output restart must be different files")
    if output.exists():
        raise FileExistsError(output)
    if factor < 1 or factor & (factor - 1):
        raise ValueError("refinement factor must be a positive power of two")
    if not cells or any(cell < 0 for cell in cells):
        raise ValueError("at least one non-negative target cell is required")
    if chunk_particles < 1 or chunk_particles > 0xFFFFFFFF:
        raise ValueError("invalid output chunk size")
    if not source.is_file():
        raise FileNotFoundError(source)

    source_hash_before = sha256(source)
    before, source_ids, source_rng = scan_restart(source, legacy_parcel_mass)
    missing = sorted(cells - set(before.counts))
    if missing:
        raise ValueError(f"target cells are empty or absent: {missing}")
    selected_source_particles = sum(before.counts[cell] for cell in cells)
    extra_particles = selected_source_particles * (factor - 1)
    output_particles = before.particle_count + extra_particles

    maximum_id = int(np.max(source_ids)) if len(source_ids) else -1
    if maximum_id + extra_particles > np.iinfo(np.uint64).max:
        raise OverflowError("not enough uint64 origId space for refinement")
    new_ids = np.arange(
        maximum_id + 1,
        maximum_id + 1 + extra_particles,
        dtype=np.uint64,
    )
    new_rng = splitmix64(new_ids ^ np.uint64(seed))
    if len(np.intersect1d(source_rng, new_rng, assume_unique=True)):
        raise ValueError("derived clone RNG collides with an existing RNG state; change seed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        write_ugkp_restart(
            temporary,
            _refined_chunks(
                source,
                legacy_parcel_mass,
                set(cells),
                factor,
                new_ids,
                new_rng,
            ),
            output_particles,
            chunk_particles,
        )
        after, output_ids, output_rng = scan_restart(temporary, None)
        if len(output_ids) != output_particles or len(output_rng) != output_particles:
            raise ValueError("weighted restart reread count mismatch")
        maximum_residual, conservation_rows = compare_stats(
            before, after, set(cells), factor, tolerance
        )
        if sha256(source) != source_hash_before:
            raise RuntimeError("source restart changed during offline refinement")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    capacity = None
    if properties_path is not None:
        capacity = update_capacity(
            Path(properties_path),
            output_particles,
            capacity_reserve_fraction,
            capacity_reserve_minimum,
        )

    if manifest_path is None:
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    else:
        manifest_path = Path(manifest_path)
    selected_csv = manifest_path.with_suffix(".selected_cells.csv")
    conservation_csv = manifest_path.with_suffix(".conservation.csv")

    selected_lines = ["cell,source_particles,output_particles"]
    for cell in sorted(cells):
        selected_lines.append(
            f"{cell},{before.counts[cell]},{after.counts[cell]}"
        )
    _atomic_text(selected_csv, "\n".join(selected_lines) + "\n")

    fieldnames = ("cell", "invariant", "before", "after", "residual", "scaled_residual")
    csv_lines = [",".join(fieldnames)]
    for row in conservation_rows:
        csv_lines.append(",".join(str(row[name]) for name in fieldnames))
    _atomic_text(conservation_csv, "\n".join(csv_lines) + "\n")

    manifest: dict[str, object] = {
        "format": "UGKP_PARTICLES_SCHEMA1_BIN",
        "source": str(source),
        "source_sha256": source_hash_before,
        "output": str(output),
        "output_sha256": sha256(output),
        "source_particles": before.particle_count,
        "output_particles": output_particles,
        "selected_cells": sorted(cells),
        "selected_source_particles": selected_source_particles,
        "factor": factor,
        "seed": int(seed),
        "chunk_particles": chunk_particles,
        "injection_mass_changed": False,
        "capacity": capacity,
        "conservation_tolerance": tolerance,
        "maximum_scaled_conservation_residual": maximum_residual,
        "conservation_passed": True,
        "selected_cells_csv": str(selected_csv),
        "conservation_csv": str(conservation_csv),
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell", type=int, action="append", required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--legacy-parcel-mass", type=float)
    parser.add_argument("--seed", type=int, default=30020260811)
    parser.add_argument("--chunk-particles", type=int, default=262144)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--capacity-reserve-fraction", type=float, default=0.05)
    parser.add_argument("--capacity-reserve-minimum", type=int, default=100000)
    parser.add_argument("--tolerance", type=float, default=5.0e-13)
    args = parser.parse_args()
    manifest = refine_restart(
        args.source,
        args.output,
        set(args.cell),
        args.factor,
        args.legacy_parcel_mass,
        args.seed,
        args.chunk_particles,
        args.manifest,
        args.properties,
        args.capacity_reserve_fraction,
        args.capacity_reserve_minimum,
        args.tolerance,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
