from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from particle_restart_io import (
    ParticleRestartError,
    binary_chunks,
    payload_size,
    read_header,
    sha256,
    validate_particle_mass,
)


FLOAT_FIELDS = ("px", "py", "pz", "pux", "puy", "puz", "pT", "pTheta", "pd")
WEIGHTED_FLOAT_FIELDS = FLOAT_FIELDS + ("pm",)
INT_FIELDS = ("cell", "status")
UINT_FIELDS = ("rng", "orig_id")
ALL_FIELDS = WEIGHTED_FLOAT_FIELDS + INT_FIELDS + UINT_FIELDS

UNWEIGHTED_SPECS = ([(name, "<f8", 1) for name in FLOAT_FIELDS]
                    + [(name, "<i4", 1) for name in INT_FIELDS]
                    + [(name, "<u8", 1) for name in UINT_FIELDS])
WEIGHTED_SPECS = UNWEIGHTED_SPECS[:len(FLOAT_FIELDS)] + [("pm", "<f8", 1)] + UNWEIGHTED_SPECS[len(FLOAT_FIELDS):]
GAS_MARKER = re.compile(r"UGKP_PARTICLES_SCHEMA[0-9]+(?:_BIN)?")


def field_specs(stream, path, marker, total, bound):
    if not GAS_MARKER.fullmatch(marker):
        raise ParticleRestartError(f"{path}: cannot decode particle field layout {marker!r}")
    if bound is None:
        if marker.endswith("_BIN"):
            raise ParticleRestartError(f"{path}: missing particle chunk bound")
        return [(name, "ascii", width) for name, _, width in UNWEIGHTED_SPECS]
    if not marker.endswith("_BIN"):
        raise ParticleRestartError(f"{path}: text particle layout has an unexpected chunk bound")
    extent = payload_size(stream)
    for specs in (WEIGHTED_SPECS, UNWEIGHTED_SPECS):
        field_bytes = sum(np.dtype(dtype).itemsize for _, dtype, _ in specs)
        chunk_bytes = extent - total * field_bytes
        if 4 * bool(total) <= chunk_bytes <= 4 * total and chunk_bytes % 4 == 0:
            return specs
    raise ParticleRestartError(f"{path}: binary payload cannot supply the declared particle fields")


def _legacy_mass(mass, path):
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ParticleRestartError(f"{path}: missing field pm; a finite positive legacy parcel mass is required")
    return float(mass)


def _validate_chunk(chunk, path):
    missing = [name for name in ALL_FIELDS if name not in chunk]
    if missing:
        raise ParticleRestartError(f"{path}: missing particle fields {', '.join(missing)}")
    if len({len(chunk[name]) for name in ALL_FIELDS}) != 1:
        raise ParticleRestartError(f"{path}: particle fields have inconsistent lengths")
    validate_particle_mass(chunk["pm"], path)


def _text_chunks(stream, path, total, legacy_mass, chunk_particles):
    if chunk_particles < 1:
        raise ValueError("text particle chunk size must be positive")
    rows = []
    done = 0
    for line in stream:
        if not line.strip():
            continue
        row = line.split()
        if done >= total:
            raise ParticleRestartError(f"{path}: extra text particle records after particle {total}")
        if len(row) != len(UNWEIGHTED_SPECS):
            raise ParticleRestartError(f"{path}: incomplete text fields at particle {done}, expected 13")
        rows.append(row)
        done += 1
        if len(rows) == chunk_particles:
            yield _text_rows(rows, legacy_mass, path, done - len(rows))
            rows = []
    if done != total:
        raise ParticleRestartError(f"{path}: expected {total} particles, found {done}")
    if rows:
        yield _text_rows(rows, legacy_mass, path, done - len(rows))


def _text_rows(rows, legacy_mass, path, first):
    data = {}
    for column, (name, dtype, _) in enumerate(UNWEIGHTED_SPECS):
        try:
            data[name] = np.asarray([row[column] for row in rows], dtype=dtype)
        except (ValueError, OverflowError) as error:
            raise ParticleRestartError(f"{path}: invalid text particle field {name} in chunk at particle {first}") from error
    data["pm"] = np.full(len(rows), legacy_mass, dtype="<f8")
    return data


def read_chunks(stream, path, header, legacy_parcel_mass, text_chunk_particles):
    marker, total, bound = header
    specs = field_specs(stream, path, marker, total, bound)
    weighted = specs is WEIGHTED_SPECS
    mass = _legacy_mass(legacy_parcel_mass, path) if total and not weighted else None
    if bound is None:
        yield from _text_chunks(stream, path, total, mass, text_chunk_particles)
        return
    for chunk in binary_chunks(stream, path, total, specs):
        if not weighted:
            chunk["pm"] = np.full(len(chunk["px"]), mass, dtype="<f8")
        validate_particle_mass(chunk["pm"], path)
        yield chunk


def iter_restart_chunks(
    path: Path,
    legacy_parcel_mass: float | None = None,
    text_chunk_particles: int = 262144,
) -> Iterator[dict[str, np.ndarray]]:
    path = Path(path)
    with path.open("rb") as stream:
        yield from read_chunks(stream, path, read_header(stream, path), legacy_parcel_mass, text_chunk_particles)


def write_ugkp_restart(
    path: Path,
    chunks: Iterable[dict[str, np.ndarray]],
    total_particles: int,
    chunk_particles: int = 262144,
) -> None:
    path = Path(path)
    if total_particles < 0:
        raise ValueError("total particle count must be non-negative")
    if chunk_particles < 1 or chunk_particles > 0xFFFFFFFF:
        raise ValueError("invalid output chunk size")
    emitted = 0
    with path.open("xb") as stream:
        stream.write(
            f"UGKP_PARTICLES_SCHEMA1_BIN {total_particles} {chunk_particles}\n".encode("ascii")
        )
        for input_chunk in chunks:
            _validate_chunk(input_chunk, path)
            input_count = len(input_chunk["pm"])
            for start in range(0, input_count, chunk_particles):
                count = min(chunk_particles, input_count - start)
                stream.write(struct.pack("<I", count))
                stop = start + count
                for name in WEIGHTED_FLOAT_FIELDS:
                    np.ascontiguousarray(input_chunk[name][start:stop], dtype="<f8").tofile(stream)
                for name in INT_FIELDS:
                    np.ascontiguousarray(input_chunk[name][start:stop], dtype="<i4").tofile(stream)
                for name in UINT_FIELDS:
                    np.ascontiguousarray(input_chunk[name][start:stop], dtype="<u8").tofile(stream)
                emitted += count
        if emitted != total_particles:
            raise ValueError(
                f"output declared {total_particles} particles but wrote {emitted}"
            )
        stream.flush()
        os.fsync(stream.fileno())


def collect_restart(
    path: Path,
    legacy_parcel_mass: float | None = None,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in ALL_FIELDS}
    for chunk in iter_restart_chunks(path, legacy_parcel_mass):
        for name in ALL_FIELDS:
            pieces[name].append(chunk[name])
    result: dict[str, np.ndarray] = {}
    for name in ALL_FIELDS:
        dtype = "<f8" if name in WEIGHTED_FLOAT_FIELDS else "<i4" if name in INT_FIELDS else "<u8"
        result[name] = np.concatenate(pieces[name]) if pieces[name] else np.empty(0, dtype=dtype)
    return result
