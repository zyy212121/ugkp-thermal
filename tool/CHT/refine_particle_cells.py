#!/usr/bin/env python3
                                                                           

from __future__ import annotations

import argparse
from functools import partial
import json
import sys
from pathlib import Path

from ugkp_particle_restart import (
    FLOAT_FIELDS,
    iter_restart_chunks,
    sha256,
    write_ugkp_restart,
    CHT_MARKER,
    FSH_MARKER,
    input_metadata,
    legacy_fsh_layout,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "restart"))
from particle_refinement import (
    INVARIANT_NAMES,
    FSH_INVARIANT_NAMES,
    SnapshotStats,
    compare_stats,
    splitmix64,
    update_capacity,
    refine_restart as _refine_restart,
)


def refine_restart(
    source: Path,
    output: Path,
    cells: set[int],
    factor: int,
    legacy_parcel_mass: float | None,
    seed: int,
    chunk_particles: int = 262144,
    manifest_path: Path | None = None,
    properties_path: Path | None = None,
    capacity_reserve_fraction: float = 0.05,
    capacity_reserve_minimum: int = 100000,
    tolerance: float = 5.0e-13,
    *,
    preserve_fsh_layout: bool = False,
) -> dict[str, object]:
    source_metadata = input_metadata(source)
    output_marker = (FSH_MARKER if preserve_fsh_layout and
                     source_metadata["source_format"].endswith("_FP64_FIELDS_BIN") else CHT_MARKER)
    return _refine_restart(
        source, output, cells, factor, legacy_parcel_mass, seed,
        chunk_particles, manifest_path, properties_path,
        capacity_reserve_fraction, capacity_reserve_minimum, tolerance,
        reader=partial(iter_restart_chunks, output_marker=output_marker),
        writer=partial(write_ugkp_restart, output_marker=output_marker),
        output_marker=output_marker, source_metadata=source_metadata,
        invariants=FSH_INVARIANT_NAMES if output_marker == FSH_MARKER else INVARIANT_NAMES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell", type=int, action="append", required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--legacy-parcel-mass", type=float)
    parser.add_argument("--seed", type=int, default=30020260811)
    parser.add_argument("--chunk-particles", type=int, default=262144)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--capacity-reserve-fraction", type=float, default=0.05)
    parser.add_argument("--capacity-reserve-minimum", type=int, default=100000)
    parser.add_argument("--tolerance", type=float, default=5.0e-13)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        source_header = stream.readline().decode("ascii", errors="replace").split()
    if source_header and legacy_fsh_layout(source_header[0]):
        fsh_tools = Path(__file__).resolve().parents[2] / "legacy" / "tool" / "FSH"
        sys.path.insert(0, str(fsh_tools))
        from refine_fsh_particle_cells import refine_restart as refine_fsh_restart

        manifest = refine_fsh_restart(
            args.source,
            args.output,
            set(args.cell),
            args.factor,
            args.legacy_parcel_mass,
            args.seed,
            args.chunk_particles,
            args.manifest,
            args.properties,
            args.capacity_reserve_fraction,
            args.capacity_reserve_minimum,
            args.tolerance,
        )
        print(json.dumps(manifest, sort_keys=True))
        return
    manifest = refine_restart(
        args.source,
        args.output,
        set(args.cell),
        args.factor,
        args.legacy_parcel_mass,
        args.seed,
        args.chunk_particles,
        args.manifest,
        args.properties,
        args.capacity_reserve_fraction,
        args.capacity_reserve_minimum,
        args.tolerance,
        preserve_fsh_layout=True,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(f"Error: {error}") from None
