#!/usr/bin/env python3
"""Compare the MSS7 laminar and SST temperature histories with experiment."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parent
THERMAL = RESULTS.parent
DEFAULT_LAMINAR = THERMAL / "MSS7_laminar"
DEFAULT_SST = THERMAL / "MSS7_turbulent_wallModel"
EXPERIMENT = RESULTS / "mss7_temperature_histories_digitized.csv"
PROBE_LOCATIONS = RESULTS / "mss7_temperature_probe_locations.csv"
OUTPUT_CSV = RESULTS / "mss7_temperature_laminar_sst_experiment.csv"
OUTPUT_PNG = RESULTS / "mss7_temperature_laminar_sst_experiment.png"
TIME_TOLERANCE_S = 1.0e-8


def numeric_directories(path: Path) -> list[Path]:
    directories: list[tuple[float, Path]] = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                try:
                    directories.append((float(child.name), child))
                except ValueError:
                    pass
    return [directory for _, directory in sorted(directories)]


def read_probe_series(path: Path) -> tuple[list[tuple[float, float, float]], dict[float, list[float]]]:
    coordinates: list[tuple[float, float, float]] | None = None
    samples: dict[float, list[float]] = {}
    for directory in numeric_directories(path):
        probe_file = directory / "T"
        if not probe_file.is_file():
            continue
        local_coordinates: list[tuple[float, float, float]] = []
        for line in probe_file.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"#\s*Probe\s+\d+\s+\(([-+0-9.eE]+)\s+"
                r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",
                line,
            )
            if match:
                local_coordinates.append(tuple(float(value) for value in match.groups()))
                continue
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            samples[values[0]] = values[1:]
        if coordinates is None:
            coordinates = local_coordinates
        elif local_coordinates and local_coordinates != coordinates:
            raise RuntimeError(f"probe coordinates changed between chunks below {path}")
    if coordinates is None or not coordinates or not samples:
        raise RuntimeError(f"no temperature probe data found below {path}")
    if any(len(values) != len(coordinates) for values in samples.values()):
        raise RuntimeError(f"probe column count mismatch below {path}")
    return coordinates, samples


def read_probe_locations() -> dict[str, dict[str, str]]:
    with PROBE_LOCATIONS.open(newline="", encoding="utf-8") as stream:
        return {row["label"]: row for row in csv.DictReader(stream)}


def wall_temperature(
    coordinates: list[tuple[float, float, float]],
    values: list[float],
    target_x: float,
) -> float:
    ordered = sorted(zip((point[0] for point in coordinates), values))
    x = np.asarray([item[0] for item in ordered], dtype=float)
    temperature = np.asarray([item[1] for item in ordered], dtype=float)
    if target_x < x[0] or target_x > x[-1]:
        raise RuntimeError("the requested wall probe is not bracketed by patch probes")
    return float(np.interp(target_x, x, temperature))


def read_case(case: Path, target_wall_x: float) -> dict[float, tuple[float, float, float]]:
    post = case / "postProcessing"
    wall_coordinates, wall_samples = read_probe_series(
        post / "mss7ThroatWallProbe" / "graphite"
    )
    internal_coordinates, internal_samples = read_probe_series(
        post / "mss7ThroatTemperatureProbes" / "graphite"
    )
    if len(internal_coordinates) != 2:
        raise RuntimeError(f"expected two internal MSS7 probes in {case}")

    result: dict[float, tuple[float, float, float]] = {}
    for time_s, internal_values in internal_samples.items():
        candidates = [
            candidate
            for candidate in wall_samples
            if abs(candidate - time_s) <= TIME_TOLERANCE_S
        ]
        if not candidates:
            continue
        wall_values = wall_samples[candidates[0]]
        result[time_s] = (
            wall_temperature(wall_coordinates, wall_values, target_wall_x),
            internal_values[0],
            internal_values[1],
        )
    if not result:
        raise RuntimeError(f"no matched wall/internal MSS7 samples in {case}")
    return result


def read_experiment() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with EXPERIMENT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"no experimental data in {EXPERIMENT}")
    time = np.asarray([float(row["time_s"]) for row in rows])
    values = {
        "Tw": np.asarray([float(row["Tw_K_TTRT"]) for row in rows]),
        "Tn1": np.asarray([float(row["Tn1_K"]) for row in rows]),
        "Tn2": np.asarray([float(row["Tn2_K"]) for row in rows]),
    }
    return time, values


def match_cases(
    laminar: dict[float, tuple[float, float, float]],
    sst: dict[float, tuple[float, float, float]],
) -> list[tuple[float, tuple[float, float, float], tuple[float, float, float]]]:
    matches = []
    for time_s, laminar_values in sorted(laminar.items()):
        candidates = [candidate for candidate in sst if abs(candidate - time_s) <= TIME_TOLERANCE_S]
        if candidates:
            matches.append((time_s, laminar_values, sst[candidates[0]]))
    if not matches:
        raise RuntimeError("the laminar and SST cases have no matching probe times")
    return matches


def write_csv(
    matches: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]],
    experiment_time: np.ndarray,
    experiment: dict[str, np.ndarray],
    destination: Path,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    keys = ("Tw", "Tn1", "Tn2")
    for time_s, laminar_values, sst_values in matches:
        row: dict[str, float] = {"time_s": time_s}
        for index, key in enumerate(keys):
            row[f"{key}_experiment_K"] = float(
                np.interp(time_s, experiment_time, experiment[key])
            )
            row[f"{key}_laminar_K"] = laminar_values[index]
            row[f"{key}_sst_K"] = sst_values[index]
        rows.append(row)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot(rows: list[dict[str, float]], destination: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 18,
            "axes.labelsize": 22,
            "legend.fontsize": 14,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    time = np.asarray([row["time_s"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), facecolor="white")
    labels = (r"$T_w$", r"$T_{N1}$", r"$T_{N2}$")
    for index, (axis, key, label) in enumerate(zip(axes, ("Tw", "Tn1", "Tn2"), labels)):
        experiment = np.asarray([row[f"{key}_experiment_K"] for row in rows])
        laminar = np.asarray([row[f"{key}_laminar_K"] for row in rows])
        sst = np.asarray([row[f"{key}_sst_K"] for row in rows])
        axis.plot(time, experiment, color="black", linewidth=2.6, label="Experiment")
        axis.plot(
            time,
            sst,
            color="#C44E52",
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="SST wall model",
        )
        axis.plot(
            time,
            laminar,
            color="#4C72B0",
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Laminar",
        )
        all_values = np.concatenate((experiment, laminar, sst))
        span = max(20.0, float(all_values.max() - all_values.min()))
        time_span = max(1.0e-12, float(time[-1] - time[0]))
        axis.set_xlim(float(time[0] - 0.025 * time_span), float(time[-1] + 0.025 * time_span))
        axis.set_ylim(float(all_values.min() - 0.08 * span), float(all_values.max() + 0.08 * span))
        axis.set_xlabel(r"Time, $t$ (s)")
        axis.set_ylabel(r"Temperature, $T$ (K)")
        axis.text(0.70 if index == 0 else 0.05, 0.92, f"({chr(97 + index)}) {label}", transform=axis.transAxes)
        axis.tick_params(top=True, right=True, length=6, width=1.0)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
        if index == 0:
            axis.legend(loc="best", frameon=False)
    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(destination, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laminar-case", type=Path, default=DEFAULT_LAMINAR)
    parser.add_argument("--sst-case", type=Path, default=DEFAULT_SST)
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--png-output", type=Path, default=OUTPUT_PNG)
    args = parser.parse_args()

    locations = read_probe_locations()
    target_wall_x = float(locations["Tw"]["x_m"])
    laminar = read_case(args.laminar_case.resolve(), target_wall_x)
    sst = read_case(args.sst_case.resolve(), target_wall_x)
    experiment_time, experiment = read_experiment()
    matches = match_cases(laminar, sst)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    rows = write_csv(matches, experiment_time, experiment, args.csv_output)
    plot(rows, args.png_output)
    print(f"wrote={args.csv_output}")
    print(f"wrote={args.png_output}")
    print(f"records={len(rows)} lastTime={rows[-1]['time_s']:.12g}")


if __name__ == "__main__":
    main()
