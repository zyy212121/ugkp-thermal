"""Historical schema1..6 reference bytes for decoder tests, never solver output."""
from pathlib import Path
import os
import struct
from typing import Iterable
import numpy as np
from fsh_particle_restart_schema6 import _field_specs, _normalise, _validate_chunk, MAXIMUM_CHUNK_PARTICLES

def write_historical_fsh_restart(path: Path, chunks: Iterable[dict[str, np.ndarray]],
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
