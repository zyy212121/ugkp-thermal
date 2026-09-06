from pathlib import Path
import struct

import numpy as np


def field_layout(schema):
    if schema not in range(1, 7):
        raise ValueError(f"Unsupported FSH particle restart schema {schema}")
    time_type = "<f8" if schema == 6 else "<f4"
    fields = [(name, "<f8", 1) for name in
              ("px", "py", "pz", "ux", "uy", "uz", "T", "theta", "d", "mass")]
    fields += [("cell", "<i4", 1), ("status", "<i4", 1),
               ("rng", "<u8", 1), ("original_id", "<u8", 1),
               ("stuck", "u1", 1), ("stuck_face", "<i4", 1),
               ("contact_area", "<f4", 1)]
    if schema >= 2:
        fields += [("contact_duration", time_type, 1),
                   ("contact_maximum_area", "<f4", 1)]
    if schema >= 3:
        fields += [("contact_peak_fraction", "<f4", 1)]
    if schema >= 4:
        fields += [("cold_node_specific_enthalpy", "<f4", 8),
                   ("cold_ring_solid_mass", "<f4", 8),
                   ("cold_frozen_area", "<f4", 1),
                   ("cold_contact_age", time_type, 1)]
    if schema >= 5:
        fields += [("cold2d_node_specific_enthalpy", "<f4", 64),
                   ("cold2d_ring_contact_age", time_type, 8),
                   ("cold2d_frozen_area", "<f4", 1)]
    return fields


def iter_fsh_chunks(path):
    path = Path(path)
    with path.open("rb") as stream:
        try:
            header = stream.readline().decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError(f"{path}: non-ASCII particle restart header") from error
        schemas = {f"UGKP_FSH_PARTICLES_SCHEMA{i}_BIN": i for i in range(1, 7)}
        if len(header) != 3 or header[0] not in schemas:
            raise ValueError(f"{path}: unsupported FSH particle restart header {header}")
        total, maximum = int(header[1]), int(header[2])
        if total < 0 or maximum <= 0:
            raise ValueError(f"{path}: invalid particle count or chunk bound")
        fields = field_layout(schemas[header[0]])
        remaining_bytes = path.stat().st_size - stream.tell()
        bytes_per_particle = sum(np.dtype(dtype).itemsize * width for _, dtype, width in fields)
        if total * bytes_per_particle > remaining_bytes:
            raise ValueError(f"{path}: truncated particle restart payload")
        loaded = 0
        while loaded < total:
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError(f"{path}: truncated particle chunk count")
            count = struct.unpack("<I", raw)[0]
            if count < 1 or count > maximum or loaded + count > total:
                raise ValueError(f"{path}: invalid particle chunk size {count}")
            chunk = {}
            for name, dtype, width in fields:
                values = np.fromfile(stream, dtype=dtype, count=count * width)
                if values.size != count * width:
                    raise ValueError(f"{path}: truncated particle field {name}")
                chunk[name] = values if width == 1 else values.reshape(count, width)
            loaded += count
            yield chunk
        if stream.read(1):
            raise ValueError(f"{path}: trailing data in particle restart")
