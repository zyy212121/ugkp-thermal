#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS))
import bentsrm_case_reader as reader


THERMAL = RESULTS.parent
HOT = THERMAL / "bentSRM_hotWall"
COLD = THERMAL / "bentSRM_coldWall"
PREFIX = "bentSRM_wall_model"
PLOT_START = 1.5
PLOT_END = 2.6
TC_COLORS = ("#C44E52", "#4C72B0", "#55A868", "#DD8452")
FLUX_COLORS = ("#C44E52", "#4C72B0", "#55A868", "#DD8452", "#6B7A8F")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 22,
    "legend.fontsize": 13,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def select_case(case):
    reader.CASE = case
    reader.EXPERIMENT = case / "assets/experimental/graphite_thermocouples.csv"


def configure_axes(axis):
    axis.set_xlim(PLOT_START, PLOT_END)
    axis.grid(False)
    axis.tick_params(direction="in", top=True, right=True, length=6, width=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def markevery(values):
    return max(1, len(values)//18)


def inside_legend(axis, handles, labels, location="best", columns=2):
    return axis.legend(
        handles,
        labels,
        loc=location,
        ncol=columns,
        frameon=False,
        fontsize=13,
        handlelength=2.1,
        columnspacing=0.8,
        labelspacing=0.35,
        borderaxespad=0.5,
    )


def temperature_results():
    select_case(HOT)
    hot_time, hot_temperature = reader.read_probe_history()
    experiment_time, measured = reader.read_experiment()
    select_case(COLD)
    cold_time, cold_temperature = reader.read_probe_history()

    destination = RESULTS / f"{PREFIX}_temperature_comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wall_model", "time_s", "TC1_5mm_K", "TC2_10mm_K", "TC3_15mm_K", "TC4_20mm_K"))
        for model, times, values in (
            ("measured", experiment_time, measured),
            ("hotWall", hot_time, hot_temperature),
            ("coldWall1D", cold_time, cold_temperature),
        ):
            for time, row in zip(times, values):
                writer.writerow((model, f"{time:.17g}", *(f"{value:.12g}" for value in row)))

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    measured_mask = (experiment_time >= PLOT_START) & (experiment_time <= PLOT_END)
    hot_mask = (hot_time >= PLOT_START) & (hot_time <= PLOT_END)
    cold_mask = (cold_time >= PLOT_START) & (cold_time <= PLOT_END)
    for index in range(4):
        axis.plot(experiment_time[measured_mask], measured[measured_mask, index],
                  color=TC_COLORS[index], linestyle="-", linewidth=2.2)
        axis.plot(hot_time[hot_mask], hot_temperature[hot_mask, index],
                  color=TC_COLORS[index], linestyle="--", linewidth=1.4,
                  marker="o", markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, markevery=markevery(hot_time[hot_mask]))
        axis.plot(cold_time[cold_mask], cold_temperature[cold_mask, index],
                  color=TC_COLORS[index], linestyle="-.", linewidth=1.4,
                  marker="^", markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, markevery=markevery(cold_time[cold_mask]))
    handles = [Line2D([], [], color=color, linewidth=2.2) for color in TC_COLORS]
    labels = [f"TC{index}" for index in range(1, 5)]
    handles.extend((
        Line2D([], [], color="black", linestyle="-", linewidth=2.2),
        Line2D([], [], color="black", linestyle="--", marker="o", markerfacecolor="white", linewidth=1.4),
        Line2D([], [], color="black", linestyle="-.", marker="^", markerfacecolor="white", linewidth=1.4),
    ))
    labels.extend(("Mea.", "Hot wall", "Cold wall"))
    inside_legend(axis, handles, labels, location="best", columns=2)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Temperature $T$ (K)")
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / f"{PREFIX}_temperature_comparison.png", dpi=600, facecolor="white")
    plt.close(fig)


def heat_flux_results():
    records = []
    series = []
    for label, case in (("hotWall", HOT), ("coldWall1D", COLD)):
        select_case(case)
        times, fluxes = reader.read_wall_heat_flux_history()
        total = np.sum(fluxes, axis=1)
        values = np.column_stack((total, fluxes))
        series.append((label, times, values))
        records.extend((label, float(time), *map(float, row)) for time, row in zip(times, values))
    destination = RESULTS / f"{PREFIX}_wall_heat_flux.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wall_model", "time_s", "total_W_m2", "radiation_W_m2", "reflected_W_m2", "deposited_W_m2", "convection_W_m2"))
        writer.writerows(records)

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    for model_index, (_, times, values) in enumerate(series):
        mask = (times >= PLOT_START) & (times <= PLOT_END)
        marker = "o" if model_index == 0 else "^"
        linestyle = "--" if model_index == 0 else "-."
        for component in range(5):
            axis.plot(times[mask], values[mask, component], color=FLUX_COLORS[component],
                      linestyle=linestyle, linewidth=1.5, marker=marker,
                      markersize=4.8, markerfacecolor="white", markeredgewidth=0.9,
                      markevery=markevery(times[mask]))
    component_labels = ("Total", "Radiation", "Reflection", "Deposition", "Convection")
    handles = [Line2D([], [], color=color, linewidth=1.8) for color in FLUX_COLORS]
    labels = list(component_labels)
    handles.extend((
        Line2D([], [], color="black", linestyle="--", marker="o", markerfacecolor="white"),
        Line2D([], [], color="black", linestyle="-.", marker="^", markerfacecolor="white"),
    ))
    labels.extend(("Hot wall", "Cold wall"))
    inside_legend(axis, handles, labels, location="best", columns=2)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Wall heat flux $q''$ (W m$^{-2}$)")
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / f"{PREFIX}_wall_heat_flux.png", dpi=600, facecolor="white")
    plt.close(fig)


def sommerfeld_records(case):
    select_case(case)
    owner_normals, _ = reader.coupled_owner_normals()
    records = []
    for directory in reader.numeric_subdirectories(case):
        restart = directory / "gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        sommerfeld, normal_speed = reader.incoming_sommerfeld(restart, owner_normals)
        if sommerfeld.size:
            records.append((float(directory.name), int(sommerfeld.size), float(np.median(sommerfeld)),
                            float(np.mean(sommerfeld)), float(np.percentile(sommerfeld, 25)),
                            float(np.percentile(sommerfeld, 75)), float(np.median(normal_speed))))
        else:
            records.append((float(directory.name), 0, math.nan, math.nan, math.nan, math.nan, math.nan))
    if not records:
        raise RuntimeError(f"No particle restart writes in {case}")
    return np.asarray(records, dtype=float)


def sommerfeld_results():
    series = (("hotWall", sommerfeld_records(HOT)), ("coldWall1D", sommerfeld_records(COLD)))
    destination = RESULTS / f"{PREFIX}_coupled_wall_incoming_sommerfeld.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wall_model", "time_s", "count", "median", "mean", "p25", "p75", "normal_speed_median_m_s"))
        for label, data in series:
            for row in data:
                writer.writerow((label, *(f"{value:.17g}" for value in row)))
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    for index, (label, data) in enumerate(series):
        valid = np.isfinite(data[:, 2]) & (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
        color = ("#4C72B0", "#C44E52")[index]
        marker = ("o", "^")[index]
        axis.fill_between(data[valid, 0], data[valid, 4], data[valid, 5], color=color, alpha=0.14, linewidth=0)
        axis.plot(data[valid, 0], data[valid, 2], color=color, linewidth=1.6,
                  linestyle=("--", "-.")[index], marker=marker, markersize=5,
                  markerfacecolor="white", markeredgewidth=0.9, label=label)
    axis.axhline(20.0, color="black", linestyle="--", linewidth=1.4, label="Threshold")
    axis.set_yscale("log")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Incoming-particle Sommerfeld number $K$")
    inside_legend(axis, *axis.get_legend_handles_labels(), location="best", columns=1)
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / f"{PREFIX}_coupled_wall_incoming_sommerfeld.png", dpi=600, facecolor="white")
    plt.close(fig)


def radiating_records(case):
    select_case(case)
    _, face_areas = reader.coupled_owner_normals()
    records = []
    for directory in reader.numeric_subdirectories(case):
        restart = directory / "gpuResidentStrictParticles.dat"
        if restart.is_file():
            records.append((float(directory.name),) + reader.normalized_radiating_area(restart, face_areas))
    if not records:
        raise RuntimeError(f"No particle restart writes in {case}")
    return np.asarray(records, dtype=float)


def radiating_area_results():
    series = (("hotWall", radiating_records(HOT)), ("coldWall1D", radiating_records(COLD)))
    destination = RESULTS / f"{PREFIX}_coupled_wall_radiating_area_fraction.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wall_model", "time_s", "radiating_area_fraction", "contact_area_fraction", "effective_contact_area_m2", "coupled_wall_area_m2"))
        for label, data in series:
            for row in data:
                writer.writerow((label, *(f"{value:.17g}" for value in row)))
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    for index, (label, data) in enumerate(series):
        valid = (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
        axis.plot(data[valid, 0], data[valid, 1], color=("#4C72B0", "#C44E52")[index],
                  linestyle=("--", "-.")[index], linewidth=1.6,
                  marker=("o", "^")[index], markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, label=label)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Normalized radiating area $1-A_c/A_w$")
    axis.set_ylim(0.0, 1.02)
    inside_legend(axis, *axis.get_legend_handles_labels(), location="best", columns=1)
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / f"{PREFIX}_coupled_wall_radiating_area_fraction.png", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    temperature_results()
    heat_flux_results()
    sommerfeld_results()
    radiating_area_results()
    print(f"wrote={RESULTS}")


if __name__ == "__main__":
    main()
