"""Read current owner-specific particle layouts, without guessing old widths.

Historical schema7 files are decoded by thermal_particle_restart_schema7.py.
CHT schema9 retains only its two float enthalpy profiles; FSH schema8 is FP64.
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
from thermal_particle_restart_schema7 import field_specs as legacy_specs

CHT_MARKER = "UGKP_CHT_PARTICLES_SCHEMA9_FLOAT_COLD_PROFILES_BIN"
FSH_MARKER = "UGKP_FSH_PARTICLES_SCHEMA8_FP64_FIELDS_BIN"
LEGACY_CHT_MARKER = "UGKP_CHT_PARTICLES_SCHEMA8_THERMAL_CARRY_BIN"
PROFILES = {"pColdNodeSpecificEnthalpy", "pCold2DNodeSpecificEnthalpy"}


def field_specs(marker):
    # Dispatch identifies bytes, never a solver precision or admission policy.
    if marker == LEGACY_CHT_MARKER:
        return legacy_specs("UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN") + [("pThermalEnthalpyResidual", "<f8", 1)]
    if marker not in (CHT_MARKER, FSH_MARKER):
        return legacy_specs(marker)
    specs = [(name, "<f4" if marker == CHT_MARKER and name in PROFILES else
              "<f8" if dtype in ("<f4", "<f8") else dtype, width)
             for name, dtype, width in legacy_specs("UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN")]
    if marker == CHT_MARKER:
        specs.append(("pThermalEnthalpyResidual", "<f8", 1))
    return specs


def iter_chunks(path):
    with Path(path).open("rb") as stream:
        header = stream.readline(256)
        if not header.endswith(b"\n"):
            raise ValueError("invalid particle restart header terminator or length")
        tokens = header.decode("ascii").split()
        if len(tokens) != 3:
            raise ValueError("invalid particle restart header")
        marker = tokens[0]
        specs = field_specs(marker)
        total, bound = map(int, tokens[1:])
        if total < 0:
            raise ValueError("negative particle restart count")
        # The native reader treats the saved bound as descriptive metadata.
        done = 0
        while done < total:
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError("truncated particle restart chunk count")
            count = struct.unpack("<I", raw)[0]
            if count == 0 or count > total - done:
                raise ValueError("invalid particle restart chunk size")
            data = {}
            for name, dtype, width in specs:
                values = np.fromfile(stream, dtype=dtype, count=count*width)
                if len(values) != count*width:
                    raise ValueError("truncated particle restart field " + name)
                data[name] = values if width == 1 else values.reshape(count, width)
            done += count
            yield marker, data


def normalize(chunk, marker):
    """Fill absent legacy fields with the native loader's existing defaults.

    Missing saved velocities are zero, not recovered impact history. The CHT
    residual is specific enthalpy and is copied unchanged during refinement.
    """
    count = len(chunk["pm"])
    result = {}
    for name, dtype, width in field_specs(marker):
        if name in chunk:
            result[name] = np.asarray(chunk[name], dtype=dtype)
        else:
            shape = count if width == 1 else (count, width)
            result[name] = np.full(shape, -1 if name == "pStuckFaceId" else 0, dtype=dtype)
    return result


def write_restart(path, chunks, total, chunk_particles, *, cht=False):
    """Write the current owner layout; historical formats are input only."""
    import os
    marker = CHT_MARKER if cht else FSH_MARKER
    specs = field_specs(marker)
    emitted = 0
    with Path(path).open("xb") as stream:
        stream.write(f"{marker} {total} {chunk_particles}\n".encode("ascii"))
        for chunk in chunks:
            chunk = normalize(chunk, marker)
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
