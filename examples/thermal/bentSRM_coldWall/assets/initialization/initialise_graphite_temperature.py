#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


CASE = Path(__file__).resolve().parents[2]
DATA = CASE / "assets/experimental/graphite_thermocouples.csv"
INTERFACE_POINT = (0.0, -0.005974642177, 0.436025357800)
DEPTH_DIRECTION = (0.0, -1.0/math.sqrt(2.0), 1.0/math.sqrt(2.0))


def measured_profile(time_s: float):
    with DATA.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = min(rows, key=lambda item: abs(float(item["time_s"]) - time_s))
    if abs(float(row["time_s"]) - time_s) > 1e-10:
        raise RuntimeError(f"measurement time {time_s} is absent")
    t5 = float(row["TC1_5mm_K"])
    t10 = float(row["TC2_10mm_K"])
    return (
        (0.0, 2.0*t5 - t10),
        (0.005, t5),
        (0.010, t10),
        (0.015, float(row["TC3_15mm_K"])),
        (0.020, float(row["TC4_20mm_K"])),
        (0.025, float(row["TC5_25mm_rear_wall_K"])),
    )


def interpolate(profile, depth: float) -> float:
    depth = min(max(depth, profile[0][0]), profile[-1][0])
    for left, right in zip(profile, profile[1:]):
        if depth <= right[0]:
            f = (depth-left[0])/(right[0]-left[0])
            return left[1] + f*(right[1]-left[1])
    return profile[-1][1]


def read_centres(path: Path):
    text = path.read_text()
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise RuntimeError(f"cannot parse {path}")
    result = [tuple(map(float, item.split())) for item in re.findall(r"\(([^)]+)\)", match.group(2))]
    if len(result) != int(match.group(1)):
        raise RuntimeError("cell-centre count mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=1.5)
    parser.add_argument("--directory", type=Path, default=CASE/"1.5/graphite")
    args = parser.parse_args()
    profile = measured_profile(args.time)
    centre_path = args.directory/"C"
    centres = read_centres(centre_path)
    values = []
    for point in centres:
        depth = sum((point[i]-INTERFACE_POINT[i])*DEPTH_DIRECTION[i] for i in range(3))
        values.append(interpolate(profile, depth))
    path = args.directory/"T"
    source = path.read_text()
    replacement = "internalField nonuniform List<scalar>\n{}\n(\n{}\n);".format(len(values), "\n".join(f"{x:.17g}" for x in values))
    source, count = re.subn(r"internalField\s+(?:uniform\s+[-+0-9.eE]+|nonuniform\s+List<scalar>\s+\d+\s*\(.*?\))\s*;", replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("cannot replace graphite internal field")
    source = re.sub(r"(graphite_to_fluid\s*\{.*?value\s+uniform\s+)[-+0-9.eE]+", rf"\g<1>{profile[0][1]:.17g}", source, flags=re.S)
    source = re.sub(r"(graphiteBottom\s*\{.*?value\s+uniform\s+)[-+0-9.eE]+", rf"\g<1>{profile[-1][1]:.17g}", source, flags=re.S)
    path.write_text(source)
    centre_path.unlink()
    for component in ("Cx", "Cy", "Cz"):
        component_path = args.directory/component
        if component_path.exists():
            component_path.unlink()
    print(f"cells={len(values)} Tmin={min(values):.6f} Tmax={max(values):.6f} interface={profile[0][1]:.6f}")


if __name__ == "__main__":
    main()
