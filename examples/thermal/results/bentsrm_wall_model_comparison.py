#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS))
import bentsrm_case_reader as reader
import single_alumina_drop_validation as single_drop


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


def marker_spacing(reference_values):
    values = np.asarray(reference_values, dtype=float)
    if values.size < 2:
        return max(PLOT_END - PLOT_START, 1.0)
    return max((values[-1] - values[0])/18.0, np.finfo(float).eps)


def markevery(values, spacing):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    first_target = PLOT_START + math.ceil(
        (values[0] - PLOT_START)/spacing - 1.0e-12
    )*spacing
    targets = np.arange(first_target, values[-1] + 0.5*spacing, spacing)
    indices = np.searchsorted(values, targets, side="left")
    indices = indices[indices < values.size]
    return np.unique(indices).tolist()


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
    hot_probe_time, hot_probe_temperature = reader.read_probe_history()
    hot_time, _, hot_minimum, hot_maximum = (
        reader.read_graphite_depth_band_history()
    )
    hot_temperature = np.column_stack([
        np.interp(hot_time, hot_probe_time, hot_probe_temperature[:, index])
        for index in range(4)
    ])
    experiment_time, measured = reader.read_experiment()
    select_case(COLD)
    cold_probe_time, cold_probe_temperature = reader.read_probe_history()
    cold_time, _, cold_minimum, cold_maximum = (
        reader.read_graphite_depth_band_history()
    )
    cold_temperature = np.column_stack([
        np.interp(cold_time, cold_probe_time, cold_probe_temperature[:, index])
        for index in range(4)
    ])

    destination = RESULTS / f"{PREFIX}_temperature_comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["wall_model", "time_s"]
        for index, depth in enumerate((5, 10, 15, 20), 1):
            header.extend((
                f"TC{index}_{depth}mm_K",
                f"TC{index}_{depth}mm_band_min_K",
                f"TC{index}_{depth}mm_band_max_K",
            ))
        writer.writerow(header)
        for model, times, values, minima, maxima in (
            ("measured", experiment_time, measured, measured, measured),
            ("hotWall", hot_time, hot_temperature, hot_minimum, hot_maximum),
            ("coldWall1D", cold_time, cold_temperature, cold_minimum, cold_maximum),
        ):
            for row_index, time in enumerate(times):
                row = [model, f"{time:.17g}"]
                for column in range(4):
                    row.extend((
                        f"{values[row_index, column]:.12g}",
                        f"{minima[row_index, column]:.12g}",
                        f"{maxima[row_index, column]:.12g}",
                    ))
                writer.writerow(row)

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    measured_mask = (experiment_time >= PLOT_START) & (experiment_time <= PLOT_END)
    hot_mask = (hot_time >= PLOT_START) & (hot_time <= PLOT_END)
    cold_mask = (cold_time >= PLOT_START) & (cold_time <= PLOT_END)
    hot_plot_time = hot_time[hot_mask]
    cold_plot_time = cold_time[cold_mask]
    spacing = marker_spacing(hot_plot_time)
    for index in range(4):
        axis.plot(experiment_time[measured_mask], measured[measured_mask, index],
                  color=TC_COLORS[index], linestyle="-", linewidth=2.2)
        axis.fill_between(
            hot_plot_time,
            hot_minimum[hot_mask, index],
            hot_maximum[hot_mask, index],
            color=TC_COLORS[index], alpha=0.12, linewidth=0,
        )
        axis.plot(hot_plot_time, hot_temperature[hot_mask, index],
                  color=TC_COLORS[index], linestyle="--", linewidth=1.4,
                  marker="o", markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, markevery=markevery(hot_plot_time, spacing))
        axis.fill_between(
            cold_plot_time,
            cold_minimum[cold_mask, index],
            cold_maximum[cold_mask, index],
            color=TC_COLORS[index], alpha=0.06, linewidth=0,
        )
        axis.plot(cold_plot_time, cold_temperature[cold_mask, index],
                  color=TC_COLORS[index], linestyle="-.", linewidth=1.4,
                  marker="^", markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, markevery=markevery(cold_plot_time, spacing))
    handles = [Line2D([], [], color=color, linewidth=2.2) for color in TC_COLORS]
    labels = [f"TC{index}" for index in range(1, 5)]
    handles.extend((
        Line2D([], [], color="black", linestyle="-", linewidth=2.2),
        Line2D([], [], color="black", linestyle="--", marker="o", markerfacecolor="white", linewidth=1.4),
        Line2D([], [], color="black", linestyle="-.", marker="^", markerfacecolor="white", linewidth=1.4),
        Patch(facecolor="gray", edgecolor="none", alpha=0.12),
    ))
    labels.extend(("Mea.", "Hot wall", "Cold wall", r"Calc. $\pm2$ mm"))
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
    hot_times = series[0][1]
    hot_mask = (hot_times >= PLOT_START) & (hot_times <= PLOT_END)
    spacing = marker_spacing(hot_times[hot_mask])
    for model_index, (_, times, values) in enumerate(series):
        mask = (times >= PLOT_START) & (times <= PLOT_END)
        marker = "o" if model_index == 0 else "^"
        linestyle = "--" if model_index == 0 else "-."
        for component in range(5):
            axis.plot(times[mask], values[mask, component], color=FLUX_COLORS[component],
                      linestyle=linestyle, linewidth=1.5, marker=marker,
                      markersize=4.8, markerfacecolor="white", markeredgewidth=0.9,
                      markevery=markevery(times[mask], spacing))
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
    hot_valid = (
        np.isfinite(series[0][1][:, 2])
        & (series[0][1][:, 0] >= PLOT_START)
        & (series[0][1][:, 0] <= PLOT_END)
    )
    spacing = marker_spacing(series[0][1][hot_valid, 0])
    for index, (label, data) in enumerate(series):
        valid = np.isfinite(data[:, 2]) & (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
        color = ("#4C72B0", "#C44E52")[index]
        marker = ("o", "^")[index]
        axis.fill_between(data[valid, 0], data[valid, 4], data[valid, 5], color=color, alpha=0.14, linewidth=0)
        axis.plot(data[valid, 0], data[valid, 2], color=color, linewidth=1.6,
                  linestyle=("--", "-.")[index], marker=marker, markersize=5,
                  markerfacecolor="white", markeredgewidth=0.9,
                  markevery=markevery(data[valid, 0], spacing), label=label)
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
    hot_valid = (
        (series[0][1][:, 0] >= PLOT_START)
        & (series[0][1][:, 0] <= PLOT_END)
    )
    spacing = marker_spacing(series[0][1][hot_valid, 0])
    for index, (label, data) in enumerate(series):
        valid = (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
        axis.plot(data[valid, 0], data[valid, 1], color=("#4C72B0", "#C44E52")[index],
                  linestyle=("--", "-.")[index], linewidth=1.6,
                  marker=("o", "^")[index], markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, markevery=markevery(data[valid, 0], spacing),
                  label=label)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Normalized radiating area $1-A_c/A_w$")
    axis.set_ylim(0.0, 1.02)
    inside_legend(axis, *axis.get_legend_handles_labels(), location="best", columns=1)
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(RESULTS / f"{PREFIX}_coupled_wall_radiating_area_fraction.png", dpi=600, facecolor="white")
    plt.close(fig)


def coupled_wall_particle_temperature_records(case, cold_wall):
    select_case(case)
    _, face_areas = reader.coupled_owner_normals()
    coupled_faces = np.asarray(tuple(face_areas), dtype=np.int64)
    properties = (
        single_drop.read_cold_wall_thermal_properties(case)
        if cold_wall else None
    )
    records = []
    for directory in reader.numeric_subdirectories(case):
        restart = directory / "gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        direct_temperature = []
        bottom_temperature = []
        top_temperature = []
        enthalpy_mean_temperature = []
        for chunk in reader.particle_chunks(restart):
            selected = (
                (chunk["stuck"] != 0)
                & np.isin(chunk["stuck_face"], coupled_faces)
            )
            if not np.any(selected):
                continue
            if not cold_wall:
                values = chunk["T"][selected]
                direct_temperature.extend(values[np.isfinite(values)].tolist())
                continue
            enthalpy = chunk["cold_node_specific_enthalpy"][selected]
            active = np.all(np.isfinite(enthalpy) & (enthalpy > 0.0), axis=1)
            if not np.all(active):
                raise RuntimeError(
                    f"Cold-wall contact particles lack an active 8-node enthalpy profile in {restart}"
                )
            bottom_temperature.extend(
                single_drop.cold_wall_temperature_from_specific_enthalpy_k(
                    float(value), properties
                )
                for value in enthalpy[:, 0]
            )
            top_temperature.extend(
                single_drop.cold_wall_temperature_from_specific_enthalpy_k(
                    float(value), properties
                )
                for value in enthalpy[:, -1]
            )
            enthalpy_mean_temperature.extend(
                single_drop.cold_wall_temperature_from_specific_enthalpy_k(
                    float(value), properties
                )
                for value in np.mean(enthalpy, axis=1)
            )
        time = float(directory.name)
        if cold_wall:
            for component, values in (
                ("bottomLayer", bottom_temperature),
                ("topLayer", top_temperature),
                ("enthalpyMean", enthalpy_mean_temperature),
            ):
                values = np.asarray(values, dtype=float)
                if values.size:
                    records.append((
                        time, component, int(values.size),
                        float(np.median(values)), float(np.mean(values)),
                        float(np.percentile(values, 25)),
                        float(np.percentile(values, 75)),
                    ))
        else:
            values = np.asarray(direct_temperature, dtype=float)
            if values.size:
                records.append((
                    time, "particle", int(values.size),
                    float(np.median(values)), float(np.mean(values)),
                    float(np.percentile(values, 25)),
                    float(np.percentile(values, 75)),
                ))
    return records, properties


def coupled_wall_particle_temperature_results():
    hot_records, _ = coupled_wall_particle_temperature_records(HOT, False)
    cold_records, properties = coupled_wall_particle_temperature_records(COLD, True)
    if not hot_records and not cold_records:
        raise RuntimeError("No coupled-wall contact-particle temperatures are available")
    destination = RESULTS / f"{PREFIX}_coupled_wall_particle_temperature.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "wall_model", "temperature_component", "time_s", "particle_count",
            "temperature_median_K", "temperature_mean_K",
            "temperature_p25_K", "temperature_p75_K",
        ))
        for model, records in (("hotWall", hot_records), ("coldWall", cold_records)):
            for time, component, count, median, mean, p25, p75 in records:
                writer.writerow((
                    model, component, f"{time:.17g}", count,
                    f"{median:.12g}", f"{mean:.12g}",
                    f"{p25:.12g}", f"{p75:.12g}",
                ))

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    hot_times = np.asarray([row[0] for row in hot_records], dtype=float)
    hot_window = hot_times[(hot_times >= PLOT_START) & (hot_times <= PLOT_END)]
    spacing = marker_spacing(hot_window)
    specifications = (
        ("Hot wall", hot_records, "particle", "#4C72B0", "--", "o"),
        ("Cold wall bottom layer", cold_records, "bottomLayer", "#C44E52", "-.", "^"),
        ("Cold wall top (gas-side) layer", cold_records, "topLayer", "#DD8452", (0, (5, 2)), "D"),
        ("Cold wall enthalpy mean", cold_records, "enthalpyMean", "#55A868", ":", "s"),
    )
    for label, records, component, color, linestyle, marker in specifications:
        selected = [row for row in records if row[1] == component]
        if not selected:
            continue
        data = np.asarray(
            [(row[0], row[3], row[5], row[6]) for row in selected],
            dtype=float,
        )
        visible = (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
        data = data[visible]
        if data.size == 0:
            continue
        axis.fill_between(
            data[:, 0], data[:, 2], data[:, 3], color=color,
            alpha=0.12, linewidth=0,
        )
        axis.plot(
            data[:, 0], data[:, 1], color=color, linestyle=linestyle,
            linewidth=1.6, marker=marker, markersize=5,
            markerfacecolor="white", markeredgewidth=0.9,
            markevery=markevery(data[:, 0], spacing), label=label,
        )
    if properties is not None:
        solidification_temperature = properties.melting_temperature_k
        axis.axhline(
            solidification_temperature, color="black", linestyle="--",
            linewidth=1.2,
            label=f"Solidification temperature ({solidification_temperature:.0f} K)",
        )
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Wall-contact particle temperature $T_p$ (K)")
    inside_legend(
        axis, *axis.get_legend_handles_labels(),
        location="center right", columns=1,
    )
    configure_axes(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(
        RESULTS / f"{PREFIX}_coupled_wall_particle_temperature.png",
        dpi=600, facecolor="white",
    )
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    temperature_results()
    heat_flux_results()
    sommerfeld_results()
    radiating_area_results()
    coupled_wall_particle_temperature_results()
    print(f"wrote={RESULTS}")


if __name__ == "__main__":
    main()
