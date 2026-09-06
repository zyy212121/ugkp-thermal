#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


FLOAT_FIELDS = ("px", "py", "pz", "pux", "puy", "puz", "pT", "pTheta", "pd")
WEIGHTED_FLOAT_FIELDS = FLOAT_FIELDS + ("pm",)
INT_FIELDS = ("cell", "status")
UINT_FIELDS = ("rng", "orig_id")
FSH_UINT8_FIELDS = ("pStuck",)
FSH_INT_FIELDS = ("pStuckFaceId",)
FSH_FLOAT32_FIELDS = ("pDepositionArea", "pContactMaximumArea", "pContactPeakFraction")
FSH_TIME_FIELDS = ("pContactDuration", "pColdContactAge", "pCold2DRingContactAge")
FSH_COLD_FLOAT32_FIELDS = (
    ("pColdNodeSpecificEnthalpy", 8),
    ("pColdRingSolidMass", 8),
    ("pColdFrozenArea", 1),
    ("pCold2DNodeSpecificEnthalpy", 64),
    ("pCold2DFrozenArea", 1),
)
ALL_FIELDS = (
    WEIGHTED_FLOAT_FIELDS + INT_FIELDS + UINT_FIELDS + FSH_UINT8_FIELDS
    + FSH_INT_FIELDS + FSH_FLOAT32_FIELDS + FSH_TIME_FIELDS
    + tuple(name for name, _ in FSH_COLD_FLOAT32_FIELDS)
)
SCHEMA_NAMES = {f"UGKP_FSH_PARTICLES_SCHEMA{i}_BIN": i for i in range(1, 7)}
MAXIMUM_CHUNK_PARTICLES = 262144


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_specs(schema: int) -> tuple[tuple[str, str, int], ...]:
    if schema not in range(1, 7):
        raise ValueError(f"unsupported FSH particle restart schema {schema}")
    specs = [(name, "<f8", 1) for name in WEIGHTED_FLOAT_FIELDS]
    specs += [(name, "<i4", 1) for name in INT_FIELDS]
    specs += [(name, "<u8", 1) for name in UINT_FIELDS]
    specs += [("pStuck", "<u1", 1), ("pStuckFaceId", "<i4", 1),
              ("pDepositionArea", "<f4", 1)]
    time_dtype = "<f8" if schema == 6 else "<f4"
    if schema >= 2:
        specs += [("pContactDuration", time_dtype, 1), ("pContactMaximumArea", "<f4", 1)]
    if schema >= 3:
        specs += [("pContactPeakFraction", "<f4", 1)]
    if schema >= 4:
        specs += [("pColdNodeSpecificEnthalpy", "<f4", 8),
                  ("pColdRingSolidMass", "<f4", 8),
                  ("pColdFrozenArea", "<f4", 1),
                  ("pColdContactAge", time_dtype, 1)]
    if schema >= 5:
        specs += [("pCold2DNodeSpecificEnthalpy", "<f4", 64),
                  ("pCold2DRingContactAge", time_dtype, 8),
                  ("pCold2DFrozenArea", "<f4", 1)]
    return tuple(specs)


def restart_schema(path: Path) -> int:
    with Path(path).open("rb") as stream:
        header = stream.readline().decode("ascii").split()
    if not header or header[0] not in SCHEMA_NAMES:
        raise ValueError(f"{path}: unsupported FSH particle restart header {header!r}")
    return SCHEMA_NAMES[header[0]]


def _read_array(stream, dtype: str, count: int, width: int, name: str, path: Path) -> np.ndarray:
    values = np.fromfile(stream, dtype=dtype, count=count * width)
    if len(values) != count * width:
        raise ValueError(f"{path}: truncated {name} array")
    return values if width == 1 else values.reshape(count, width)


def _zero(dtype: str, count: int, width: int) -> np.ndarray:
    return np.zeros(count if width == 1 else (count, width), dtype=dtype)


def _normalise(chunk: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    result = dict(chunk)
    specs = {name: (dtype, width) for name, dtype, width in _field_specs(6)}
    for name in ALL_FIELDS:
        if name not in result:
            dtype, width = specs[name]
            result[name] = _zero(dtype, count, width)
    return result


def _validate_chunk(chunk: dict[str, np.ndarray], path: Path) -> None:
    missing = [name for name in ALL_FIELDS if name not in chunk]
    if missing:
        raise ValueError(f"{path}: missing particle fields {missing}")
    count = len(chunk["pm"])
    if any(len(chunk[name]) != count for name in ALL_FIELDS):
        raise ValueError(f"{path}: particle fields have inconsistent lengths")
    if not np.all(np.isfinite(chunk["pm"])) or np.any(chunk["pm"] <= 0):
        raise ValueError(f"{path}: particle statistical mass must be finite and positive")
    if np.any(chunk["pStuck"] > 3):
        raise ValueError(f"{path}: pStuck must be in the range 0..3")
    area = chunk["pDepositionArea"]
    face = chunk["pStuckFaceId"]
    stuck = chunk["pStuck"] != 0
    if not np.all(np.isfinite(area)) or np.any(area < 0):
        raise ValueError(f"{path}: invalid deposition area")
    contact_duration = chunk["pContactDuration"]
    contact_maximum_area = chunk["pContactMaximumArea"]
    contact_peak_fraction = chunk["pContactPeakFraction"]
    if (
        not np.all(np.isfinite(contact_duration))
        or not np.all(np.isfinite(contact_maximum_area))
        or not np.all(np.isfinite(contact_peak_fraction))
        or np.any(contact_duration < 0)
        or np.any(contact_maximum_area < 0)
        or np.any(contact_peak_fraction < 0)
        or np.any(contact_peak_fraction > 1)
    ):
        raise ValueError(f"{path}: invalid wall contact metadata")
    if np.any((~stuck) & ((face != -1) | (area != 0))):
        raise ValueError(f"{path}: mobile FSH parcel carries deposition state")
    if np.any(face < -2) or np.any(stuck & (face == -1)):
        raise ValueError(f"{path}: stuck FSH parcel has invalid wall-face state")
    if np.any((face == -2) & (area != 0)):
        raise ValueError(f"{path}: unresolved wall-face sentinel carries an area")
    deposited = chunk["pStuck"] == 1
    valid_deposited_contact = (
        (
            (contact_duration == 0)
            & (contact_maximum_area == 0)
            & (contact_peak_fraction == 0)
        )
        | (
            (contact_duration > 0)
            & (contact_maximum_area > 0)
            & (contact_peak_fraction > 0)
            & (contact_peak_fraction < 1)
        )
    )
    if np.any(deposited & ~valid_deposited_contact):
        raise ValueError(f"{path}: deposited parcel has invalid wall contact metadata")
    transient = (chunk["pStuck"] == 2) | (chunk["pStuck"] == 3)
    if np.any(transient & (contact_duration <= 0)):
        raise ValueError(f"{path}: transient wall parcel has no contact duration")
    if np.any(transient & (contact_maximum_area <= 0)):
        raise ValueError(f"{path}: transient wall parcel has no maximum contact area")
    if np.any(transient & ((contact_peak_fraction <= 0) | (contact_peak_fraction >= 1))):
        raise ValueError(f"{path}: transient wall parcel has invalid peak fraction")
    if np.any(transient & (chunk["pTheta"] > contact_duration)):
        raise ValueError(f"{path}: transient wall parcel exceeds contact duration")
    for name in ALL_FIELDS:
        if name not in ("cell", "status", "rng", "orig_id", "pStuck", "pStuckFaceId"):
            if not np.all(np.isfinite(chunk[name])):
                raise ValueError(f"{path}: non-finite particle field {name}")


def iter_restart_chunks(path: Path, legacy_parcel_mass: float | None = None,
                        text_chunk_particles: int = 262144) -> Iterator[dict[str, np.ndarray]]:
    del legacy_parcel_mass, text_chunk_particles
    path = Path(path)
    with path.open("rb") as stream:
        try:
            header = stream.readline().decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError(f"{path}: invalid restart header") from error
        if not header or header[0] not in SCHEMA_NAMES or len(header) != 3:
            raise ValueError(f"{path}: unsupported particle restart {header!r}")
        schema = SCHEMA_NAMES[header[0]]
        total, maximum = int(header[1]), int(header[2])
        if total < 0 or maximum <= 0 or maximum > MAXIMUM_CHUNK_PARTICLES:
            raise ValueError(f"{path}: invalid binary restart sizes")
        specs = _field_specs(schema)
        emitted = 0
        while emitted < total:
            raw_count = stream.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"{path}: truncated binary chunk count")
            count = struct.unpack("<I", raw_count)[0]
            if count == 0 or count > maximum or count > total - emitted:
                raise ValueError(f"{path}: invalid binary chunk size {count}")
            chunk = {name: _read_array(stream, dtype, count, width, name, path)
                     for name, dtype, width in specs}
            chunk = _normalise(chunk, count)
            _validate_chunk(chunk, path)
            emitted += count
            yield chunk
        if stream.read(1):
            raise ValueError(f"{path}: trailing data after final particle chunk")


def write_fsh_restart(path: Path, chunks: Iterable[dict[str, np.ndarray]],
                      total_particles: int, chunk_particles: int = 262144,
                      schema: int = 1) -> None:
    path = Path(path)
    if (total_particles < 0 or chunk_particles < 1
            or chunk_particles > MAXIMUM_CHUNK_PARTICLES):
        raise ValueError("invalid particle count or chunk size")
    specs = _field_specs(schema)
    emitted = 0
    with path.open("xb") as stream:
        stream.write(f"UGKP_FSH_PARTICLES_SCHEMA{schema}_BIN {total_particles} {chunk_particles}\n".encode("ascii"))
        for input_chunk in chunks:
            input_count = len(input_chunk["pm"])
            input_chunk = _normalise(input_chunk, input_count)
            _validate_chunk(input_chunk, path)
            for start in range(0, input_count, chunk_particles):
                count = min(chunk_particles, input_count - start)
                stream.write(struct.pack("<I", count))
                stop = start + count
                for name, dtype, _ in specs:
                    np.ascontiguousarray(input_chunk[name][start:stop], dtype=dtype).tofile(stream)
                emitted += count
        if emitted != total_particles:
            raise ValueError(f"output declared {total_particles} particles but wrote {emitted}")
        stream.flush()
        os.fsync(stream.fileno())


def collect_restart(path: Path, legacy_parcel_mass: float | None = None) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in ALL_FIELDS}
    for chunk in iter_restart_chunks(path, legacy_parcel_mass):
        for name in ALL_FIELDS:
            pieces[name].append(chunk[name])
    specs = {name: (dtype, width) for name, dtype, width in _field_specs(6)}
    result: dict[str, np.ndarray] = {}
    for name in ALL_FIELDS:
        dtype, width = specs[name]
        result[name] = np.concatenate(pieces[name], axis=0) if pieces[name] else _zero(dtype, 0, width)
    return result
