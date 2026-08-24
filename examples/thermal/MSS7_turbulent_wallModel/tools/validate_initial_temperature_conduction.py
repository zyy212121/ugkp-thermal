#!/usr/bin/env python3


from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


CASE = Path(__file__).resolve().parents[1]
INITIALISER = CASE / "tools" / "initialise_graphite_temperature.py"
DIGITIZED = CASE.parent / "results" / "mss7_temperature_histories_digitized.csv"
R_INNER_M = 0.0118
SOLID_DEPTH_M = 0.0877
RHO_KG_M3 = 1850.0
DX_M = 1.0e-4
DT_S = 1.0e-3


def load_initialiser():
    spec = importlib.util.spec_from_file_location("mss7_initialiser", INITIALISER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {INITIALISER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_piecewise_linear_profile(depth_m: float) -> float:
    depths = np.asarray((0.0, 0.010, 0.020, 0.030))
    temperatures = np.asarray((893.8356164383562, 390.0, 335.0, 300.0))
    return float(np.interp(depth_m, depths, temperatures))


def material_properties(temperature_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cp = 651.0 * np.log(np.clip(temperature_k, 200.0, 3000.0)) - 2877.0
    conductivity = 3712.0 * np.clip(temperature_k, 200.0, 2500.0) ** -0.6
    return cp, conductivity


def digitized_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DIGITIZED.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([float(row["time_s"]) for row in rows]),
        np.asarray([float(row["Tw_K_TTRT"]) for row in rows]),
        np.asarray([float(row["Tn1_K"]) for row in rows]),
    )


def solve(profile, wall_time, wall_temperature) -> dict[float, float]:
    depth = np.arange(0.0, SOLID_DEPTH_M + 0.5 * DX_M, DX_M)
    radius = R_INNER_M + depth
    temperature = np.asarray([profile(value) for value in depth])
    output: dict[float, float] = {}
    cell_count = len(temperature)

    for step in range(1, 301):
        time_s = 1.0 + step * DT_S
        cp, conductivity = material_properties(temperature)
        face_conductivity = 0.5 * (conductivity[:-1] + conductivity[1:])
        lower = np.zeros(cell_count)
        diagonal = np.ones(cell_count)
        upper = np.zeros(cell_count)
        right_hand_side = temperature.copy()

        right_hand_side[0] = float(np.interp(time_s, wall_time, wall_temperature))
        for index in range(1, cell_count - 1):
            west_radius = radius[index] - 0.5 * DX_M
            east_radius = radius[index] + 0.5 * DX_M
            west = (
                DT_S
                * west_radius
                * face_conductivity[index - 1]
                / (RHO_KG_M3 * cp[index] * radius[index] * DX_M**2)
            )
            east = (
                DT_S
                * east_radius
                * face_conductivity[index]
                / (RHO_KG_M3 * cp[index] * radius[index] * DX_M**2)
            )
            lower[index] = -west
            diagonal[index] = 1.0 + west + east
            upper[index] = -east
        right_hand_side[-1] = 300.0

        for index in range(1, cell_count):
            factor = lower[index] / diagonal[index - 1]
            diagonal[index] -= factor * upper[index - 1]
            right_hand_side[index] -= factor * right_hand_side[index - 1]
        temperature[-1] = right_hand_side[-1] / diagonal[-1]
        for index in range(cell_count - 2, -1, -1):
            temperature[index] = (
                right_hand_side[index] - upper[index] * temperature[index + 1]
            ) / diagonal[index]

        if step in (100, 200, 300):
            output[time_s] = float(np.interp(0.010, depth, temperature))
    return output


def main() -> None:
    initialiser = load_initialiser()
    wall_time, wall_temperature, tn1_temperature = digitized_tables()
    old = solve(old_piecewise_linear_profile, wall_time, wall_temperature)
    new = solve(
        initialiser.measured_depth_temperature, wall_time, wall_temperature
    )

    old_errors: list[float] = []
    new_errors: list[float] = []
    print("time_s old_Tn1_K new_Tn1_K digitized_Tn1_K old_abs_error_K new_abs_error_K")
    for time_s in sorted(old):
        target = float(np.interp(time_s, wall_time, tn1_temperature))
        old_error = abs(old[time_s] - target)
        new_error = abs(new[time_s] - target)
        old_errors.append(old_error)
        new_errors.append(new_error)
        print(
            f"{time_s:.1f} {old[time_s]:.6f} {new[time_s]:.6f} "
            f"{target:.6f} {old_error:.6f} {new_error:.6f}"
        )

    old_max = max(old_errors)
    new_max = max(new_errors)
    reduction = 1.0 - new_max / old_max
    print(
        f"oldMaxErrorK={old_max:.6f} newMaxErrorK={new_max:.6f} "
        f"maxErrorReduction={100.0 * reduction:.2f}%"
    )
    if new_max >= 15.0 or reduction < 0.5:
        raise SystemExit("initial-temperature conduction acceptance failed")


if __name__ == "__main__":
    main()
