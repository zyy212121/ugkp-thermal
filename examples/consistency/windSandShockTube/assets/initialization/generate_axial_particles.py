#!/usr/bin/env python3
   
                                                                        
                                               

                                                                            
                                                                                
                                                                     

                                                                    
                                             
                                                                                  
                                                                                         
                                                                       
                                                                     
                                                                           
                                
                                                                
   

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
RESTART = ROOT / "0" / "gpuResidentStrictParticles.dat"

HEADER = "UGKP_PARTICLES_SCHEMA4"
LENGTH = 1.0
LEFT_STATE = {"rho": 1.0, "p": 1.0, "Tp": 1.0}
RIGHT_STATE = {"rho": 0.125, "p": 0.1, "Tp": 0.8}
SOLID_APPARENT_DENSITY = 0.5
DIAMETER = 1.0
DEFAULT_N_CELLS = 100
DEFAULT_COUNT_PER_CELL = 2000
DEFAULT_SEED = 303
GRANULAR_THETA = 0.75 / SOLID_APPARENT_DENSITY         


def axial_standard_samples(count_per_cell: int) -> np.ndarray:
    quantiles = (np.arange(count_per_cell) + 0.5) / count_per_cell
    samples = norm.ppf(quantiles)
    samples -= samples.mean()
    samples /= samples.std(ddof=0)
    return samples


def base_particles(
    n_cells: int,
    count_per_cell: int,
    length: float,
    seed: int,
) -> list[tuple]:
    dx = length / n_cells
    standard = axial_standard_samples(count_per_cell)
    ux_scale = math.sqrt(GRANULAR_THETA)
    x_centres = (np.arange(n_cells) + 0.5) * dx
    tp = np.where(x_centres <= 0.5, LEFT_STATE["Tp"], RIGHT_STATE["Tp"])

    rows: list[tuple] = []
    original_id = 0
    for cell_id in range(n_cells):
        for local_id in range(count_per_cell):
            fraction = (local_id + 0.5) / count_per_cell
            px = (cell_id + fraction) * dx
            ux = ux_scale * standard[local_id]
            rng = (seed + 0x9E3779B97F4A7C15 * (original_id + 1)) & ((1 << 64) - 1)
            rows.append(
                (
                    px,              
                    0.5,             
                    0.5,             
                    ux,               
                    0.0,              
                    0.0,              
                    float(tp[cell_id]),      
                    0.0,                 
                    DIAMETER,        
                    cell_id,              
                    1,                    
                    rng,               
                    original_id,          
                )
            )
            original_id += 1
    return rows


def write_particle_restart(rows: list[tuple]) -> None:
    RESTART.parent.mkdir(parents=True, exist_ok=True)
    with RESTART.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{HEADER} {len(rows)}\n")
        for row in rows:
            if len(row) != 13:
                raise ValueError("SCHEMA4 requires 13 columns")
            state = [f"{float(v):.17g}" for v in row[:9]]
            identifiers = [str(int(v)) for v in row[9:]]
            stream.write(" ".join(state + identifiers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gpuResidentStrictParticles.dat for windSandShockTube",
    )
    parser.add_argument(
        "--n-cells", type=int, default=DEFAULT_N_CELLS,
        help=f"axial cell count (default {DEFAULT_N_CELLS})",
    )
    parser.add_argument(
        "--count-per-cell", type=int, default=DEFAULT_COUNT_PER_CELL,
        help=f"parcels per cell (default {DEFAULT_COUNT_PER_CELL})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"deterministic rng seed (default {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if args.n_cells <= 0 or args.count_per_cell <= 0:
        raise SystemExit("n-cells and count-per-cell must be positive")

    rows = base_particles(
        args.n_cells, args.count_per_cell, LENGTH, args.seed,
    )
    write_particle_restart(rows)

    total = args.n_cells * args.count_per_cell
    parcel_mass = SOLID_APPARENT_DENSITY * (LENGTH / args.n_cells) / args.count_per_cell
    capacity_hint = int(math.ceil(total * 1.1))
    print(
        f"wrote {RESTART.relative_to(ROOT)}  "
        f"nCells={args.n_cells} countPerCell={args.count_per_cell} "
        f"totalParticles={total}  parcel_mass={parcel_mass:.17g}  "
        f"suggested particleCapacity={capacity_hint}"
    )


if __name__ == "__main__":
    main()
