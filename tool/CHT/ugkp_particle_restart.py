#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "restart"))
import gas_particle_restart as gas
from gas_particle_restart import FLOAT_FIELDS, INT_FIELDS, UINT_FIELDS, WEIGHTED_FLOAT_FIELDS
from particle_restart_io import ParticleRestartError, binary_chunks, read_header, sha256, validate_particle_mass
from thermal_particle_restart_current import CHT_MARKER, FSH_MARKER, field_specs, normalize, write_restart


ALL_FIELDS = tuple(name for name, _, _ in field_specs(CHT_MARKER))
HISTORICAL_FSH = re.compile(r"UGKP_FSH_PARTICLES_SCHEMA([0-9]+)_BIN")


def legacy_fsh_layout(marker):
    match = HISTORICAL_FSH.fullmatch(marker)
    return int(match[1]) if match and 1 <= int(match[1]) <= 6 else None


def _thermal_specs(marker, path):
    historical = legacy_fsh_layout(marker)
    if historical is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy/tool/FSH"))
        from fsh_particle_restart_schema6 import _field_specs
        return _field_specs(historical)
    try:
        return field_specs(marker)
    except ParticleRestartError as error:
        raise ParticleRestartError(f"{path}: {error}") from error


def iter_restart_chunks(path, legacy_parcel_mass=None, text_chunk_particles=262144, *, output_marker=CHT_MARKER):
    path = Path(path)
    with path.open("rb") as stream:
        header = read_header(stream, path)
        marker, total, bound = header
        if gas.GAS_MARKER.fullmatch(marker):
            for chunk in gas.read_chunks(stream, path, header, legacy_parcel_mass, text_chunk_particles):
                yield normalize(chunk, output_marker)
            return
        if bound is None:
            raise ParticleRestartError(f"{path}: missing particle chunk bound")
        historical = legacy_fsh_layout(marker)
        for chunk in binary_chunks(stream, path, total, _thermal_specs(marker, path), allow_trailer=historical is None):
            validate_particle_mass(chunk["pm"], path)
            yield normalize(chunk, output_marker)


def write_ugkp_restart(path, chunks, total_particles, chunk_particles=262144, *, output_marker=CHT_MARKER):
    write_restart(path, chunks, total_particles, chunk_particles, cht=output_marker == CHT_MARKER)


def collect_restart(path, legacy_parcel_mass=None):
    pieces = list(iter_restart_chunks(path, legacy_parcel_mass))
    return {name: np.concatenate([p[name] for p in pieces], axis=0) if pieces else np.empty((0,) if width == 1 else (0, width), dtype=dtype)
            for name, dtype, width in field_specs(CHT_MARKER)}


def input_metadata(path):
    with Path(path).open("rb") as stream:
        marker, total, bound = read_header(stream, path)
        specs = (gas.field_specs(stream, path, marker, total, bound) if gas.GAS_MARKER.fullmatch(marker)
                 else _thermal_specs(marker, path))
    dtypes = {name: dtype for name, dtype, _ in specs}
    return {"source_format": marker, "source_field_dtypes": dtypes,
            "defaulted_saved_velocity": "puxOld" not in dtypes,
            "missing_saved_velocity_default": "existing native zero initialization; history not reconstructed"}
