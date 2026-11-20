from __future__ import annotations

import hashlib
import os
import struct

import numpy as np


class ParticleRestartError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(stream, path):
    header = stream.readline(256)
    if not header.endswith(b"\n"):
        raise ParticleRestartError(f"{path}: missing particle restart header or newline")
    try:
        tokens = header.decode("ascii").split()
        total = int(tokens[1])
        bound = int(tokens[2]) if len(tokens) == 3 else None
    except (ValueError, IndexError) as error:
        raise ParticleRestartError(f"{path}: missing or invalid particle header fields") from error
    if len(tokens) not in (2, 3) or total < 0:
        raise ParticleRestartError(f"{path}: invalid particle header fields")
    return tokens[0], total, bound


def payload_size(stream):
    return os.fstat(stream.fileno()).st_size - stream.tell()


def binary_chunks(stream, path, total, specs, *, allow_trailer=False):
    remaining_bytes = payload_size(stream)
    specs = [(name, np.dtype(dtype), width) for name, dtype, width in specs]
    done = 0
    while done < total:
        raw = stream.read(4)
        if len(raw) != 4:
            raise ParticleRestartError(f"{path}: truncated chunk count at particle {done}")
        remaining_bytes -= 4
        count, = struct.unpack("<I", raw)
        if not 0 < count <= total - done:
            raise ParticleRestartError(f"{path}: invalid chunk count {count} at particle {done}")
        data = {}
        for name, dtype, width in specs:
            size = count * width
            remaining_bytes -= size * dtype.itemsize
            if remaining_bytes < 0:
                raise ParticleRestartError(f"{path}: truncated particle field {name} at particle {done}")
            values = np.fromfile(stream, dtype=dtype, count=size)
            if len(values) != size:
                raise ParticleRestartError(f"{path}: truncated particle field {name} at particle {done}")
            data[name] = values if width == 1 else values.reshape(count, width)
        done += count
        yield data
    if not allow_trailer and stream.read(1):
        raise ParticleRestartError(f"{path}: trailing bytes after the last particle chunk")


def validate_particle_mass(mass, path):
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ParticleRestartError(f"{path}: particle field pm must be finite and positive")
