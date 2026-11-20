from __future__ import annotations
import re
import struct
from pathlib import Path
import numpy as np
from thermal_particle_restart_schema7 import field_specs as legacy_specs
from particle_restart_io import ParticleRestartError, binary_chunks, read_header, validate_particle_mass
from gas_particle_restart import ALL_FIELDS as REQUIRED_FIELDS

CHT_MARKER = "UGKP_CHT_PARTICLES_SCHEMA9_FLOAT_COLD_PROFILES_BIN"
SINGLE_STATE_CHT_MARKER = "UGKP_CHT_PARTICLES_SCHEMA10_FP32_SINGLE_STATE_BIN"
FSH_MARKER = "UGKP_FSH_PARTICLES_SCHEMA8_FP64_FIELDS_BIN"
LEGACY_CHT_MARKER = "UGKP_CHT_PARTICLES_SCHEMA8_THERMAL_CARRY_BIN"
PROFILES = {"pColdNodeSpecificEnthalpy", "pCold2DNodeSpecificEnthalpy"}


def field_specs(marker):

    match = re.fullmatch(r"UGKP_(?:.+_)?PARTICLES_SCHEMA[0-9]+_(.+)_BIN", marker)
    layout = match[1] if match else ""
    if layout in ("T32", "T64"):
        return legacy_specs(f"UGKP_THERMAL_PARTICLES_SCHEMA7_{layout}_BIN")
    if layout == "THERMAL_CARRY":
        return legacy_specs("UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN") + [("pThermalEnthalpyResidual", "<f8", 1)]
    float_profiles = set()
    if layout in ("FLOAT_COLD_PROFILES", "FP32_SINGLE_STATE", "NATIVE32_RAW_COLD"):
        float_profiles = PROFILES
    elif layout == "NATIVE64_RAW_COLD":
        float_profiles = {"pCold2DNodeSpecificEnthalpy"}
    elif layout != "FP64_FIELDS":
        raise ParticleRestartError(f"cannot decode particle field layout {marker}")
    specs = [(name, "<f4" if name in float_profiles else
              "<f8" if dtype in ("<f4", "<f8") else dtype, width)
             for name, dtype, width in legacy_specs("UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN")]
    if layout in ("FLOAT_COLD_PROFILES", "NATIVE64_RAW_COLD"):
        specs.append(("pThermalEnthalpyResidual", "<f8", 1))
    return specs


def iter_chunks(path):
    with Path(path).open("rb") as stream:
        marker, total, bound = read_header(stream, path)
        if bound is None:
            raise ParticleRestartError(f"{path}: missing particle chunk bound")
        for data in binary_chunks(stream, path, total, field_specs(marker), allow_trailer=True):
            validate_particle_mass(data["pm"], path)
            yield marker, data


def normalize(chunk, marker):
    missing = [name for name in REQUIRED_FIELDS if name not in chunk]
    if missing:
        raise ParticleRestartError("missing particle fields " + ", ".join(missing))
    count = len(chunk["pm"])
    result = {}
    for name, dtype, width in field_specs(marker):
        shape = (count,) if width == 1 else (count, width)
        if name in chunk:
            values = np.asarray(chunk[name], dtype=dtype)
            if values.shape != shape:
                raise ParticleRestartError(f"particle field {name} requires shape {shape}, got {values.shape}")
            result[name] = values
        else:
            result[name] = np.full(shape, -1 if name == "pStuckFaceId" else 0, dtype=dtype)
    return result


def write_restart(path, chunks, total, chunk_particles, *, cht=False):
    import os
    marker = CHT_MARKER if cht else FSH_MARKER
    specs = field_specs(marker)
    if total < 0 or not 0 < chunk_particles <= 0xFFFFFFFF:
        raise ParticleRestartError(f"{path}: invalid output particle count or chunk size")
    emitted = 0
    with Path(path).open("xb") as stream:
        stream.write(f"{marker} {total} {chunk_particles}\n".encode("ascii"))
        for chunk in chunks:
            chunk = normalize(chunk, marker)
            validate_particle_mass(chunk["pm"], path)
            size = len(chunk["pm"])
            for start in range(0, size, chunk_particles):
                count = min(chunk_particles, size - start)
                stream.write(struct.pack("<I", count))
                for name, dtype, width in specs:
                    np.ascontiguousarray(chunk[name][start:start+count], dtype=dtype).tofile(stream)
                emitted += count
        if emitted != total:
            raise ValueError(f"output declared {total} particles but wrote {emitted}")
        stream.flush()
        os.fsync(stream.fileno())
