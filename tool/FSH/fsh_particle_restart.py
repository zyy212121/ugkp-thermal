"""FSH current restart IO with historical input support."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "restart"))
from thermal_particle_restart_current import FSH_MARKER, field_specs, iter_chunks, normalize, write_restart
from fsh_particle_restart_schema6 import iter_restart_chunks as iter_legacy_chunks, sha256, _field_specs, SCHEMA_NAMES, FLOAT_FIELDS

ALL_FIELDS = tuple(n for n, _, _ in field_specs(FSH_MARKER))

def restart_marker(path):
    with Path(path).open("rb") as stream:
        return stream.readline().decode("ascii").split()[0]

def iter_restart_chunks(path, legacy_parcel_mass=None):
    marker = restart_marker(path)
    chunks = iter_legacy_chunks(path, legacy_parcel_mass) if marker.startswith("UGKP_FSH_PARTICLES_SCHEMA") and marker != FSH_MARKER else (data for _, data in iter_chunks(path))
    for chunk in chunks:
        yield normalize(chunk, FSH_MARKER)

def write_fsh_restart(path, chunks, total_particles, chunk_particles=262144):
    write_restart(path, chunks, total_particles, chunk_particles, cht=False)

def collect_restart(path, legacy_parcel_mass=None):
    pieces = list(iter_restart_chunks(path, legacy_parcel_mass))
    return {name: np.concatenate([p[name] for p in pieces], axis=0) if pieces else np.empty((0,) if width == 1 else (0, width), dtype=dtype)
            for name, dtype, width in field_specs(FSH_MARKER)}


def input_metadata(path):
    marker = restart_marker(path)
    specs = _field_specs(SCHEMA_NAMES[marker]) if marker in SCHEMA_NAMES else field_specs(marker)
    names = {name for name, _, _ in specs}
    return {"source_format": marker,
            "source_field_dtypes": {name: dtype for name, dtype, _ in specs},
            "defaulted_saved_velocity": "puxOld" not in names,
            "missing_saved_velocity_default": "existing native zero initialization; history not reconstructed"}
