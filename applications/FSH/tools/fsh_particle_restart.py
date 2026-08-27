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
FSH_FLOAT32_FIELDS = ("pDepositionArea",)
ALL_FIELDS = (
    WEIGHTED_FLOAT_FIELDS
    + INT_FIELDS
    + UINT_FIELDS
    + FSH_UINT8_FIELDS
    + FSH_INT_FIELDS
    + FSH_FLOAT32_FIELDS
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_array(stream, dtype: str, count: int, field: str, path: Path) -> np.ndarray:
    values = np.fromfile(stream, dtype=dtype, count=count)
    if len(values) != count:
        raise ValueError(f"{path}: truncated {field} array")
    return values


def _validate_chunk(chunk: dict[str, np.ndarray], path: Path) -> None:
    lengths = {len(chunk[name]) for name in ALL_FIELDS}
    if len(lengths) != 1:
        raise ValueError(f"{path}: particle fields have inconsistent lengths")
    if not np.all(np.isfinite(chunk["pm"])) or np.any(chunk["pm"] <= 0):
        raise ValueError(f"{path}: particle statistical mass must be finite and positive")
    if np.any((chunk["pStuck"] != 0) & (chunk["pStuck"] != 1)):
        raise ValueError(f"{path}: pStuck must be zero or one")
    area = chunk["pDepositionArea"]
    face = chunk["pStuckFaceId"]
    stuck = chunk["pStuck"] != 0
    if not np.all(np.isfinite(area)) or np.any(area < 0):
        raise ValueError(f"{path}: invalid deposition area")
    if np.any((~stuck) & ((face != -1) | (area != 0))):
        raise ValueError(f"{path}: mobile FSH parcel carries deposition state")
    if np.any(stuck & (face < -2)):
        raise ValueError(f"{path}: invalid stuck face id")
    if np.any(stuck & (face == -1)):
        raise ValueError(f"{path}: stuck FSH parcel has no wall-face state")
    if np.any(stuck & (face == -2) & (area != 0)):
        raise ValueError(f"{path}: legacy V2 stuck sentinel carries an area")
    if np.any(stuck & (face >= 0) & (area <= 0)):
        raise ValueError(f"{path}: resolved stuck FSH parcel has no contact area")


def iter_restart_chunks(
    path: Path,
    legacy_parcel_mass: float | None = None,
    text_chunk_particles: int = 262144,
) -> Iterator[dict[str, np.ndarray]]:
                                                                                     
    path = Path(path)
    with path.open("rb") as stream:
        try:
            header = stream.readline().decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError(f"{path}: invalid restart header") from error
        if not header:
            raise ValueError(f"{path}: empty particle restart")
        kind = header[0]
        if kind != "UGKP_FSH_PARTICLES_SCHEMA1_BIN":
            raise ValueError(f"{path}: unsupported particle restart {kind!r}")
        if len(header) != 3:
            raise ValueError(f"{path}: invalid binary restart header")
        total = int(header[1])
        maximum_chunk = int(header[2])
        if total < 0 or maximum_chunk <= 0 or maximum_chunk > 0xFFFFFFFF:
            raise ValueError(f"{path}: invalid binary restart sizes")
        emitted = 0
        while emitted < total:
            raw_count = stream.read(4)
            if len(raw_count) != 4:
                raise ValueError(f"{path}: truncated binary chunk count")
            count = struct.unpack("<I", raw_count)[0]
            if count == 0 or count > maximum_chunk or count > total - emitted:
                raise ValueError(f"{path}: invalid binary chunk size {count}")
            chunk: dict[str, np.ndarray] = {}
            for name in FLOAT_FIELDS:
                chunk[name] = _read_array(stream, "<f8", count, name, path)
            chunk["pm"] = _read_array(stream, "<f8", count, "pm", path)
            for name in INT_FIELDS:
                chunk[name] = _read_array(stream, "<i4", count, name, path)
            for name in UINT_FIELDS:
                chunk[name] = _read_array(stream, "<u8", count, name, path)
            chunk["pStuck"] = _read_array(stream, "<u1", count, "pStuck", path)
            chunk["pStuckFaceId"] = _read_array(
                stream, "<i4", count, "pStuckFaceId", path
            )
            chunk["pDepositionArea"] = _read_array(
                stream, "<f4", count, "pDepositionArea", path
            )
            _validate_chunk(chunk, path)
            emitted += count
            yield chunk
        if stream.read(1):
            raise ValueError(f"{path}: trailing data after final particle chunk")


def write_fsh_restart(
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
            f"UGKP_FSH_PARTICLES_SCHEMA1_BIN {total_particles} {chunk_particles}\n".encode("ascii")
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
                np.ascontiguousarray(
                    input_chunk["pStuck"][start:stop], dtype="<u1"
                ).tofile(stream)
                np.ascontiguousarray(
                    input_chunk["pStuckFaceId"][start:stop], dtype="<i4"
                ).tofile(stream)
                np.ascontiguousarray(
                    input_chunk["pDepositionArea"][start:stop], dtype="<f4"
                ).tofile(stream)
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
        if name in WEIGHTED_FLOAT_FIELDS:
            dtype = "<f8"
        elif name in INT_FIELDS or name in FSH_INT_FIELDS:
            dtype = "<i4"
        elif name in UINT_FIELDS:
            dtype = "<u8"
        elif name in FSH_UINT8_FIELDS:
            dtype = "<u1"
        else:
            dtype = "<f4"
        result[name] = np.concatenate(pieces[name]) if pieces[name] else np.empty(0, dtype=dtype)
    return result
