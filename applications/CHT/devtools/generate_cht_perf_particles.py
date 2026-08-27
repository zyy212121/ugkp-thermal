#!/usr/bin/env python3
                                                                                      

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


VECTOR_RE = re.compile(
    r"^\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)$"
)
MASK64 = (1 << 64) - 1
PARCEL_MASS_LIMIT = 5.0e-8


def read_cell_centres(path: Path) -> list[tuple[float, float, float]]:
                                                                               
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if "internalField" in line)
    except StopIteration as exc:
        raise ValueError(f"missing internalField in {path}") from exc

    count_index = marker + 1
    while count_index < len(lines) and not lines[count_index].strip():
        count_index += 1
    expected = int(lines[count_index].strip())

    list_index = count_index + 1
    while list_index < len(lines) and lines[list_index].strip() != "(":
        list_index += 1
    if list_index == len(lines):
        raise ValueError(f"missing vector-list opener in {path}")

    centres: list[tuple[float, float, float]] = []
    for line in lines[list_index + 1 :]:
        stripped = line.strip()
        if stripped == ")":
            break
        match = VECTOR_RE.match(stripped)
        if not match:
            raise ValueError(f"invalid centre row in {path}: {line!r}")
        centre = tuple(float(value) for value in match.groups())
        if not all(math.isfinite(value) for value in centre):
            raise ValueError(f"non-finite centre in {path}: {line!r}")
        centres.append(centre)                          

    if len(centres) != expected:
        raise ValueError(
            f"cell-centre count mismatch in {path}: expected {expected}, got {len(centres)}"
        )
    return centres


def read_cell_volumes(path: Path) -> list[float]:
                                                                               
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if "internalField" in line)
    except StopIteration as exc:
        raise ValueError(f"missing internalField in {path}") from exc

    count_index = marker + 1
    while count_index < len(lines) and not lines[count_index].strip():
        count_index += 1
    expected = int(lines[count_index].strip())

    list_index = count_index + 1
    while list_index < len(lines) and lines[list_index].strip() != "(":
        list_index += 1
    if list_index == len(lines):
        raise ValueError(f"missing scalar-list opener in {path}")

    volumes: list[float] = []
    for line in lines[list_index + 1 :]:
        stripped = line.strip()
        if stripped == ")":
            break
        value = float(stripped)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid cell volume in {path}: {line!r}")
        volumes.append(value)

    if len(volumes) != expected:
        raise ValueError(
            f"cell-volume count mismatch in {path}: expected {expected}, got {len(volumes)}"
        )
    return volumes


def splitmix64(value: int) -> int:
                                                   
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    result = (value ^ (value >> 31)) & MASK64
    return result or 1


def allocate_counts(cell_count: int, particle_count: int) -> list[int]:
    if cell_count <= 0:
        raise ValueError("cell count must be positive")
    if particle_count < cell_count:
        raise ValueError("particle count must cover every cell")
    base, remainder = divmod(particle_count, cell_count)
    return [base + (1 if cell_id < remainder else 0) for cell_id in range(cell_count)]


def validate_parcel_mass(parcel_mass: float) -> None:
    if (
        not math.isfinite(parcel_mass)
        or parcel_mass <= 0.0
        or parcel_mass >= PARCEL_MASS_LIMIT
    ):
        raise ValueError(
            f"parcel mass must satisfy 0 < parcel mass < {PARCEL_MASS_LIMIT:g}"
        )


def write_restart(
    path: Path,
    centres: list[tuple[float, float, float]],
    particle_count: int,
) -> None:
    counts = allocate_counts(len(centres), particle_count)
    lines = [f"UGKP_PARTICLES_SCHEMA3 {particle_count}"]
    original_id = 1
    for cell_id, ((x, y, z), count) in enumerate(zip(centres, counts)):
        for _ in range(count):
            rng = splitmix64(original_id)
            lines.append(
                f"{x:.17e} {y:.17e} {z:.17e} "
                "1.00000000000000000e+00 0.00000000000000000e+00 "
                "0.00000000000000000e+00 1.00000000000000000e+03 "
                "1.00000000000000000e+00 1.50000000000000004e-05 "
                f"{cell_id} 1 {rng} {original_id} -1 0.00000000000000000e+00"
            )
            original_id += 1
    if original_id - 1 != particle_count:
        raise AssertionError("internal particle-count mismatch")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_epsg_prev(
    path: Path,
    volumes: list[float],
    counts: list[int],
    parcel_mass: float,
    rho_solid: float,
) -> None:
    if len(volumes) != len(counts):
        raise ValueError("cell-volume/count size mismatch")
    validate_parcel_mass(parcel_mass)
    if not math.isfinite(rho_solid) or rho_solid <= 0:
        raise ValueError("solid density must be finite and positive")

    values = []
    for volume, count in zip(volumes, counts):
        epsilon_s = count * parcel_mass / (rho_solid * volume)
        eps_g = 1.0 - epsilon_s
        if not 0.0 < eps_g <= 1.0:
            raise ValueError(f"invalid generated gas fraction: {eps_g}")
        values.append(eps_g)
    lines = [str(len(values)), *(f"{value:.17e}" for value in values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_source_faces(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty source-residual template: {path}")
    header = lines[0].split()
    if len(header) != 2 or header[0] != "UGKP_SOURCE_RESIDUAL_SCHEMA1":
        raise ValueError(f"invalid source-residual template header: {path}")
    expected = int(header[1])
    if len(lines) != expected + 1:
        raise ValueError(
            f"source-residual template count mismatch: expected {expected}, "
            f"got {len(lines) - 1}"
        )
    source_faces = []
    for line in lines[1:]:
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(f"invalid source-residual template row: {line!r}")
        source_faces.append(int(columns[0]))
    if len(set(source_faces)) != len(source_faces):
        raise ValueError("duplicate source face in source-residual template")
    return source_faces


def write_zero_source_residual(path: Path, source_faces: list[int]) -> None:
    lines = [f"UGKP_SOURCE_RESIDUAL_SCHEMA1 {len(source_faces)}"]
    lines.extend(
        f"{face} 0.00000000000000000e+00" for face in source_faces
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-centres", type=Path, required=True)
    parser.add_argument("--cell-volumes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsg-output", type=Path, required=True)
    parser.add_argument("--source-residual-template", type=Path, required=True)
    parser.add_argument("--source-residual-output", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=28_652)
    parser.add_argument("--parcel-mass", type=float, default=5e-9)
    parser.add_argument("--rho-solid", type=float, default=2800.0)
    args = parser.parse_args()

    validate_parcel_mass(args.parcel_mass)

    centres = read_cell_centres(args.cell_centres)
    volumes = read_cell_volumes(args.cell_volumes)
    if len(centres) != len(volumes):
        raise ValueError(
            f"centre/volume size mismatch: {len(centres)} != {len(volumes)}"
        )
    counts = allocate_counts(len(centres), args.particle_count)
    write_restart(args.output, centres, args.particle_count)
    write_epsg_prev(
        args.epsg_output,
        volumes,
        counts,
        args.parcel_mass,
        args.rho_solid,
    )
    source_faces = read_source_faces(args.source_residual_template)
    write_zero_source_residual(args.source_residual_output, source_faces)
    print(
        f"wrote {args.particle_count} particles and consistent auxiliary mirrors "
        f"across {len(centres)} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
