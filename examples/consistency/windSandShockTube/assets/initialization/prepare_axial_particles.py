#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESTART = ROOT / "0" / "gpuResidentStrictParticles.dat"


def load() -> tuple[str, list[list[str]], dict[int, list[int]]]:
    lines = RESTART.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("UGKP_PARTICLES_SCHEMA4 "):
        raise SystemExit(f"unsupported restart header in {RESTART}")
    expected = int(lines[0].split()[1])
    records = [line.split() for line in lines[1:] if line.strip()]
    if len(records) != expected or any(len(record) != 13 for record in records):
        raise SystemExit("restart particle count/schema mismatch")
    by_cell: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_cell[int(record[9])].append(index)
    return lines[0], records, dict(by_cell)


def prepare() -> None:
    header, records, by_cell = load()
    for indices in by_cell.values():
        for index in indices:
            records[index][4] = "0"
            records[index][5] = "0"

    temporary = RESTART.with_suffix(RESTART.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(header + "\n")
        for record in records:
            stream.write(" ".join(record) + "\n")
    os.replace(temporary, RESTART)


def check() -> None:
    _, records, by_cell = load()
    tolerance = 5.0e-12
    for cell, indices in sorted(by_cell.items()):
        ux = [float(records[index][3]) for index in indices]
        uy = [float(records[index][4]) for index in indices]
        uz = [float(records[index][5]) for index in indices]
        mean_x = sum(ux) / len(ux)
        variance_x = sum((value - mean_x) ** 2 for value in ux) / len(ux)
        if abs(mean_x) > tolerance:
            raise SystemExit(f"cell {cell}: non-zero axial fluctuation mean")
        if not math.isclose(variance_x, 1.5, rel_tol=1.0e-12, abs_tol=tolerance):
            raise SystemExit(f"cell {cell}: incorrect axial fluctuation variance")
        if any(value != 0.0 for value in uy + uz):
            raise SystemExit(f"cell {cell}: transverse fluctuation is not zero")
    print(f"axial particle fluctuation check: PASS ({len(records)} particles)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not arguments.check:
        prepare()
    check()


if __name__ == "__main__":
    main()
