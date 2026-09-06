#!/usr/bin/env python3
"""Read-only validator for UGKP FSH particle restart schemas 5 and 6 files."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


FORMAT = "UGKP_FSH_PARTICLES_SCHEMA5_BIN"
CORE_FIELDS = (
    ("px", "<f8", 1),
    ("py", "<f8", 1),
    ("pz", "<f8", 1),
    ("pux", "<f8", 1),
    ("puy", "<f8", 1),
    ("puz", "<f8", 1),
    ("temperature", "<f8", 1),
    ("theta", "<f8", 1),
    ("diameter", "<f8", 1),
    ("mass", "<f8", 1),
    ("cell", "<i4", 1),
    ("status", "<i4", 1),
    ("rng", "<u8", 1),
    ("original_id", "<u8", 1),
    ("wall_state", "u1", 1),
    ("face", "<i4", 1),
    ("deposition_area", "<f4", 1),
    ("contact_duration", "<f4", 1),
    ("contact_maximum_area", "<f4", 1),
    ("contact_peak_fraction", "<f4", 1),
)
COLD_FIELDS = (
    ("cold_node_specific_enthalpy", "<f4", 8),
    ("cold_ring_solid_mass", "<f4", 8),
    ("cold_frozen_area", "<f4", 1),
    ("cold_contact_age", "<f4", 1),
    ("cold2d_node_specific_enthalpy", "<f4", 64),
    ("cold2d_ring_contact_age", "<f4", 8),
    ("cold2d_frozen_area", "<f4", 1),
)
STATE_NAMES = {0: "mobile", 1: "deposited", 2: "transient_rebound", 3: "transient_deposit"}


def read_array(stream, dtype: str, count: int, field: str, path: Path) -> np.ndarray:
    result = np.fromfile(stream, dtype=dtype, count=count)
    if result.size != count:
        raise ValueError(f"{path}: truncated {field}; expected {count}, got {result.size}")
    return result


def as_builtin(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def normalized_kinematic_area(age: np.ndarray, duration: np.ndarray, peak: np.ndarray) -> np.ndarray:
    theta = np.clip(age / duration, 0.0, 1.0)
    result = np.zeros(theta.shape, dtype=np.float64)
    rising = (theta > 0.0) & (theta < 1.0) & (theta <= peak)
    x = np.zeros(theta.shape, dtype=np.float64)
    x[rising] = theta[rising] / peak[rising]
    result[rising] = x[rising] * (2.0 - x[rising])
    falling = (theta > peak) & (theta < 1.0)
    y = np.zeros(theta.shape, dtype=np.float64)
    y[falling] = (theta[falling] - peak[falling]) / (1.0 - peak[falling])
    result[falling] = np.cos(0.5 * np.pi * y[falling]) ** 2
    return result


def validate(path: Path, args: argparse.Namespace) -> dict:
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    state_counts: Counter[int] = Counter()
    status_counts: Counter[int] = Counter()
    chunks = 0
    emitted = 0
    first_seen_orig: set[int] = set()
    duplicate_original_ids = 0

    example_fields = (
        "status", "wall_state", "face", "deposition_area", "contact_duration",
        "contact_maximum_area", "contact_peak_fraction", "theta", "diameter",
        "mass", "cell", "original_id",
    )

    def record(reason: str, mask: np.ndarray, fields: dict[str, np.ndarray], base: int) -> None:
        nonlocal reason_counts
        indices = np.flatnonzero(mask)
        reason_counts[reason] += int(indices.size)
        room = args.examples - len(examples[reason])
        for local in indices[: max(room, 0)]:
            row = {"index": base + int(local)}
            for name in example_fields:
                row[name] = as_builtin(fields[name][local])
            examples[reason].append(row)

    with path.open("rb") as stream:
        raw_header = stream.readline()
        try:
            header = raw_header.decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError(f"{path}: non-ASCII header") from error
        if len(header) != 3 or header[0] not in (FORMAT, "UGKP_FSH_PARTICLES_SCHEMA6_BIN"):
            raise ValueError(f"{path}: expected {FORMAT}, got {raw_header!r}")
        is_v6 = header[0] == "UGKP_FSH_PARTICLES_SCHEMA6_BIN"
        time_fields = {"contact_duration", "cold_contact_age", "cold2d_ring_contact_age"}
        wire_core = [(name, "<f8" if is_v6 and name in time_fields else dtype, width) for name,dtype,width in CORE_FIELDS]
        wire_cold = [(name, "<f8" if is_v6 and name in time_fields else dtype, width) for name,dtype,width in COLD_FIELDS]
        total = int(header[1])
        maximum_chunk = int(header[2])
        if total < 0 or maximum_chunk <= 0 or maximum_chunk > 262144:
            raise ValueError(f"{path}: invalid header sizes")

        while emitted < total:
            raw_count = stream.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"{path}: truncated chunk count at particle {emitted}")
            count = struct.unpack("<I", raw_count)[0]
            if count == 0 or count > maximum_chunk or count > total - emitted:
                raise ValueError(f"{path}: invalid chunk size {count} at particle {emitted}")
            fields = {
                name: read_array(stream, dtype, count * width, name, path)
                for name, dtype, width in wire_core
            }
            cold = {
                name: read_array(stream, dtype, count * width, name, path)
                for name, dtype, width in wire_cold
            }
            chunks += 1

            for value, occurrences in zip(*np.unique(fields["wall_state"], return_counts=True)):
                state_counts[int(value)] += int(occurrences)
            for value, occurrences in zip(*np.unique(fields["status"], return_counts=True)):
                status_counts[int(value)] += int(occurrences)

                                                                            
            for original_id in fields["original_id"]:
                item = int(original_id)
                if item in first_seen_orig:
                    duplicate_original_ids += 1
                else:
                    first_seen_orig.add(item)

            finite_particle = np.ones(count, dtype=bool)
            for name in ("px", "py", "pz", "pux", "puy", "puz", "temperature", "theta", "diameter", "mass"):
                bad = ~np.isfinite(fields[name])
                record(f"nonfinite_{name}", bad, fields, emitted)
                finite_particle &= ~bad
            record("nonpositive_diameter", ~(fields["diameter"] > 0.0), fields, emitted)
            record("nonpositive_mass", ~(fields["mass"] > 0.0), fields, emitted)
            record("theta_negative", fields["theta"] < 0.0, fields, emitted)
            record("cell_out_of_range", (fields["cell"] < -1) | (fields["cell"] >= args.n_cells), fields, emitted)
            record("status_not_0_or_1", (fields["status"] != 0) & (fields["status"] != 1), fields, emitted)

            state = fields["wall_state"]
            face = fields["face"]
            dep = fields["deposition_area"].astype(np.float64)
            duration = fields["contact_duration"].astype(np.float64)
            maximum = fields["contact_maximum_area"].astype(np.float64)
            peak = fields["contact_peak_fraction"].astype(np.float64)
            age = fields["theta"]
            active = fields["status"] != 0
            wall_bound = active & (state != 0)
            transient = active & ((state == 2) | (state == 3))

            record("wall_state_unknown", active & (state > 3), fields, emitted)
            record("face_out_of_mesh", (face < -2) | (face >= args.n_faces), fields, emitted)
            record("mobile_face_not_minus1", (state == 0) & (face != -1), fields, emitted)
            record("mobile_nonzero_deposition", (state == 0) & (dep != 0.0), fields, emitted)
            record("stuck_face_minus1", (state != 0) & (face == -1), fields, emitted)
            record(
                "diagnostic_code2_non_candidate_face",
                wall_bound & ((face < args.candidate_start) | (face >= args.candidate_end)),
                fields,
                emitted,
            )
            record("diagnostic_code3_deposited_nonpositive_area", active & (state == 1) & ~(dep > 0.0), fields, emitted)

            for name, values in (
                ("deposition_area", dep), ("contact_duration", duration),
                ("contact_maximum_area", maximum), ("contact_peak_fraction", peak),
            ):
                record(f"nonfinite_{name}", ~np.isfinite(values), fields, emitted)
            record("negative_deposition_area", dep < 0.0, fields, emitted)
            record("negative_contact_duration", duration < 0.0, fields, emitted)
            record("negative_contact_maximum_area", maximum < 0.0, fields, emitted)
            record("negative_contact_peak_fraction", peak < 0.0, fields, emitted)

            code4 = transient & (
                ~(duration > 0.0) | ~(maximum > 0.0) | ~(peak > 0.0)
                | ~(peak < 1.0) | (dep < 0.0)
            )
            record("diagnostic_code4_invalid_transient_metadata", code4, fields, emitted)
            record("transient_age_exceeds_duration", transient & (age > duration), fields, emitted)

            deposited = active & (state == 1)
            deposited_contact_all_zero = (duration == 0.0) & (maximum == 0.0) & (peak == 0.0)
            deposited_contact_valid = (duration > 0.0) & (maximum > 0.0) & (peak > 0.0) & (peak < 1.0)
            record(
                "deposited_contact_tuple_inconsistent",
                deposited & ~(deposited_contact_all_zero | deposited_contact_valid),
                fields,
                emitted,
            )

            negative_face_payload = (face < 0) & (
                (dep != 0.0) | (duration != 0.0) | (maximum != 0.0) | (peak != 0.0)
            )
            record("negative_face_has_contact_payload", negative_face_payload, fields, emitted)

            cold_frozen = cold["cold_frozen_area"]
            cold2d_frozen = cold["cold2d_frozen_area"]
            record("nonfinite_cold_frozen_area", ~np.isfinite(cold_frozen), fields, emitted)
            record("negative_cold_frozen_area", cold_frozen < 0.0, fields, emitted)
            record("nonfinite_cold_contact_age", ~np.isfinite(cold["cold_contact_age"]), fields, emitted)
            record("negative_cold_contact_age", cold["cold_contact_age"] < 0.0, fields, emitted)
            record("nonfinite_cold2d_frozen_area", ~np.isfinite(cold2d_frozen), fields, emitted)
            record("negative_cold2d_frozen_area", cold2d_frozen < 0.0, fields, emitted)
            for name in (
                "cold_node_specific_enthalpy", "cold_ring_solid_mass",
                "cold2d_node_specific_enthalpy", "cold2d_ring_contact_age",
            ):
                reshaped = cold[name].reshape(count, -1)
                record(f"nonfinite_{name}", ~np.all(np.isfinite(reshaped), axis=1), fields, emitted)
                if name != "cold_node_specific_enthalpy":
                    record(f"negative_{name}", np.any(reshaped < 0.0, axis=1), fields, emitted)

                                                                                   
            physical_area = np.zeros(count, dtype=np.float64)
            physical_area[deposited] = dep[deposited]
            valid_transient = transient & ~code4
            if np.any(valid_transient):
                kinematic = np.zeros(count, dtype=np.float64)
                kinematic[valid_transient] = maximum[valid_transient] * normalized_kinematic_area(
                    np.clip(age[valid_transient], 0.0, duration[valid_transient]),
                    duration[valid_transient],
                    peak[valid_transient],
                )
                frozen = cold_frozen.astype(np.float64) if args.candidate_mode == "coldWall1D" else (
                    cold2d_frozen.astype(np.float64) if args.candidate_mode == "coldWall2D" else np.zeros(count)
                )
                physical_area[valid_transient] = np.maximum(
                    np.maximum(kinematic[valid_transient], frozen[valid_transient]) - dep[valid_transient],
                    0.0,
                )
            contact_positive = wall_bound & (physical_area > 0.0)
            physical_mass = (np.pi / 6.0) * args.rho_solid * fields["diameter"] ** 3
            represented = physical_area * fields["mass"] / physical_mass
            code6 = contact_positive & (~np.isfinite(represented) | ~(represented > 0.0))
            record("diagnostic_code6_invalid_represented_area", code6, fields, emitted)

            emitted += count

        trailing = stream.read(1)
        if trailing:
            raise ValueError(f"{path}: trailing bytes after declared particles")
        actual_size = stream.tell()

    diagnostic_reasons = [
        "diagnostic_code2_non_candidate_face",
        "diagnostic_code3_deposited_nonpositive_area",
        "diagnostic_code4_invalid_transient_metadata",
        "wall_state_unknown",
        "diagnostic_code6_invalid_represented_area",
    ]
    return {
        "path": str(path),
        "read_only": True,
        "format": FORMAT,
        "declared_particles": total,
        "parsed_particles": emitted,
        "chunks": chunks,
        "declared_maximum_chunk": maximum_chunk,
        "file_size": os.path.getsize(path),
        "parsed_size": actual_size,
        "n_cells": args.n_cells,
        "n_faces": args.n_faces,
        "candidate_face_half_open_range": [args.candidate_start, args.candidate_end],
        "candidate_mode": args.candidate_mode,
        "rho_solid": args.rho_solid,
        "state_counts": {STATE_NAMES.get(k, str(k)): v for k, v in sorted(state_counts.items())},
        "status_counts": {str(k): v for k, v in sorted(status_counts.items())},
        "duplicate_original_ids": duplicate_original_ids,
        "diagnostic_failure_count_sum": sum(reason_counts[name] for name in diagnostic_reasons),
        "reason_counts": dict(sorted(reason_counts.items())),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--n-faces", type=int, required=True)
    parser.add_argument("--candidate-start", type=int, required=True)
    parser.add_argument("--candidate-end", type=int, required=True)
    parser.add_argument("--candidate-mode", choices=("coldWall1D", "coldWall2D", "none"), default="coldWall1D")
    parser.add_argument("--rho-solid", type=float, default=2800.0)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    if not (0 <= args.candidate_start < args.candidate_end <= args.n_faces):
        parser.error("candidate face range must lie inside [0, n_faces)")
    results = [validate(path, args) for path in args.paths]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
