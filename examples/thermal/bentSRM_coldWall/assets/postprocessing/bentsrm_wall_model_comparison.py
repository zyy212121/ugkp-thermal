#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[2]
THERMAL = CASE_ROOT.parent
RESULTS = THERMAL/"results"/CASE_ROOT.name
DATA = RESULTS/"data"
FIGURES = RESULTS/"figures"
sys.path.insert(0, str(THERMAL/"postprocessing"))
sys.path.insert(0, str(THERMAL/"singleAluminaDrop/assets/postprocessing"))
import bentsrm_case_reader as reader
import single_alumina_drop_validation as single_drop


COLD = CASE_ROOT
PREFIX = "bentSRM_wall_model"
PLOT_START = 1.5
PLOT_END = 2.6
TEMPERATURE_PLOT_START = 1.6
TC_COLORS = ("#C44E52", "#4C72B0")
FLUX_COLORS = ("#C44E52", "#4C72B0", "#55A868", "#DD8452", "#6B7A8F")

plt.rcParams.update({
    "font.family": "serif", "font.size": 18, "axes.labelsize": 22,
    "legend.fontsize": 13, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
})


def select_case():
    reader.CASE = COLD
    reader.EXPERIMENT = COLD/"assets/experimental/graphite_thermocouples.csv"


def configure_axes(axis):
    axis.set_xlim(PLOT_START, PLOT_END)
    axis.grid(False)
    axis.tick_params(direction="in", top=True, right=True, length=6, width=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def marker_spacing(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return max(PLOT_END - PLOT_START, 1.0)
    return max((values[-1] - values[0])/18.0, np.finfo(float).eps)


def markevery(values, spacing):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    first = PLOT_START + math.ceil((values[0] - PLOT_START)/spacing - 1.0e-12)*spacing
    indices = np.searchsorted(values, np.arange(first, values[-1] + 0.5*spacing, spacing))
    return np.unique(indices[indices < values.size]).tolist()


def inside_legend(axis, location="best", columns=1):
    return axis.legend(loc=location, ncol=columns, frameon=False, fontsize=13,
                       handlelength=2.1, columnspacing=0.8, labelspacing=0.35,
                       borderaxespad=0.5)


def temperature_results():
    select_case()
    time, temperature = reader.read_graphite_depth_interpolated_history((5.0, 10.0))
    experiment_time, measured = reader.read_experiment()
    measured = measured[:, :2]
    with (DATA/f"{PREFIX}_temperature_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record_type", "time_s", "TC1_temperature_K", "TC1_depth_mm",
                         "TC2_temperature_K", "TC2_depth_mm"))
        for current, values in zip(experiment_time, measured):
            writer.writerow(("measured", f"{current:.17g}", f"{values[0]:.12g}", "5",
                             f"{values[1]:.12g}", "10"))
        for current, values in zip(time, temperature):
            writer.writerow(("coldWall", f"{current:.17g}", f"{values[0]:.12g}", "5",
                             f"{values[1]:.12g}", "10"))

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    measured_mask = (experiment_time >= TEMPERATURE_PLOT_START) & (experiment_time <= PLOT_END)
    mask = (time >= TEMPERATURE_PLOT_START) & (time <= PLOT_END)
    plot_time = time[mask]
    spacing = marker_spacing(plot_time)
    for index in range(2):
        axis.plot(experiment_time[measured_mask], measured[measured_mask, index],
                  color=TC_COLORS[index], linestyle="-", linewidth=2.2,
                  label=f"TC{index + 1} measured")
        axis.plot(plot_time, temperature[mask, index], color=TC_COLORS[index],
                  linestyle="-.", linewidth=1.5, marker="^", markersize=5,
                  markerfacecolor="white", markeredgewidth=0.9,
                  markevery=markevery(plot_time, spacing),
                  label=f"TC{index + 1} calculated")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Temperature $T$ (K)")
    configure_axes(axis)
    axis.set_xlim(TEMPERATURE_PLOT_START, PLOT_END)
    inside_legend(axis, "best", 2)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES/f"{PREFIX}_temperature_comparison.png", dpi=600, facecolor="white")
    plt.close(fig)


def heat_flux_results():
    select_case()
    times, fluxes = reader.read_wall_heat_flux_history()
    total = np.sum(fluxes, axis=1)
    values = np.column_stack((total, fluxes))
    with (DATA/f"{PREFIX}_wall_heat_flux.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "total_W_m2", "radiation_W_m2", "reflected_W_m2",
                         "deposited_W_m2", "convection_W_m2"))
        for current, row in zip(times, values):
            writer.writerow((f"{current:.17g}", *(f"{value:.17g}" for value in row)))

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    mask = (times >= PLOT_START) & (times <= PLOT_END)
    spacing = marker_spacing(times[mask])
    labels = ("Total", "Radiation", "Reflection", "Deposition", "Convection")
    for index, label in enumerate(labels):
        axis.plot(times[mask], values[mask, index], color=FLUX_COLORS[index],
                  linewidth=1.5, linestyle="-.", marker="^", markersize=4.8,
                  markerfacecolor="white", markeredgewidth=0.9,
                  markevery=markevery(times[mask], spacing), label=label)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Wall heat flux $q''$ (W m$^{-2}$)")
    configure_axes(axis)
    inside_legend(axis, "best", 2)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES/f"{PREFIX}_wall_heat_flux.png", dpi=600, facecolor="white")
    plt.close(fig)


def sommerfeld_records():
    select_case()
    owner_normals, _ = reader.coupled_owner_normals()
    records = []
    for directory in reader.numeric_subdirectories(COLD):
        restart = directory/"gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        sommerfeld, normal_speed = reader.incoming_sommerfeld(restart, owner_normals)
        if sommerfeld.size:
            records.append((float(directory.name), int(sommerfeld.size),
                            float(np.median(sommerfeld)), float(np.mean(sommerfeld)),
                            float(np.percentile(sommerfeld, 25)),
                            float(np.percentile(sommerfeld, 75)),
                            float(np.median(normal_speed))))
        else:
            records.append((float(directory.name), 0, math.nan, math.nan,
                            math.nan, math.nan, math.nan))
    if not records:
        raise RuntimeError(f"No particle restart writes in {COLD}")
    return np.asarray(records, dtype=float)


def sommerfeld_results():
    data = sommerfeld_records()
    with (DATA/f"{PREFIX}_coupled_wall_incoming_sommerfeld.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "count", "median", "mean", "p25", "p75",
                         "normal_speed_median_m_s"))
        writer.writerows(tuple(f"{value:.17g}" for value in row) for row in data)
    valid = np.isfinite(data[:, 2]) & (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    axis.fill_between(data[valid, 0], data[valid, 4], data[valid, 5],
                      color="#C44E52", alpha=0.14, linewidth=0)
    axis.plot(data[valid, 0], data[valid, 2], color="#C44E52", linewidth=1.6,
              linestyle="-.", marker="^", markersize=5, markerfacecolor="white",
              markeredgewidth=0.9,
              markevery=markevery(data[valid, 0], marker_spacing(data[valid, 0])),
              label="Calculated")
    axis.axhline(20.0, color="black", linestyle="--", linewidth=1.4, label="Threshold")
    axis.set_yscale("log")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Incoming-particle Sommerfeld number $K$")
    configure_axes(axis)
    inside_legend(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES/f"{PREFIX}_coupled_wall_incoming_sommerfeld.png", dpi=600, facecolor="white")
    plt.close(fig)


def radiating_records():
    select_case()
    _, face_areas = reader.coupled_owner_normals()
    records = []
    for directory in reader.numeric_subdirectories(COLD):
        restart = directory/"gpuResidentStrictParticles.dat"
        if restart.is_file():
            records.append((float(directory.name),) + reader.normalized_radiating_area(restart, face_areas))
    if not records:
        raise RuntimeError(f"No particle restart writes in {COLD}")
    return np.asarray(records, dtype=float)


def radiating_area_results():
    data = radiating_records()
    with (DATA/f"{PREFIX}_coupled_wall_radiating_area_fraction.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "radiating_area_fraction", "contact_area_fraction",
                         "effective_contact_area_m2", "coupled_wall_area_m2"))
        writer.writerows(tuple(f"{value:.17g}" for value in row) for row in data)
    valid = (data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    axis.plot(data[valid, 0], data[valid, 1], color="#C44E52", linestyle="-.",
              linewidth=1.6, marker="^", markersize=5, markerfacecolor="white",
              markeredgewidth=0.9,
              markevery=markevery(data[valid, 0], marker_spacing(data[valid, 0])),
              label="Calculated")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Normalized radiating area $1-A_c/A_w$")
    axis.set_ylim(0.0, 1.02)
    configure_axes(axis)
    inside_legend(axis)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES/f"{PREFIX}_coupled_wall_radiating_area_fraction.png", dpi=600, facecolor="white")
    plt.close(fig)


def cold_wall_temperature_array(specific_enthalpy, properties):
    enthalpy = np.asarray(specific_enthalpy, dtype=float)
    solidus = properties.melting_temperature_k - 0.5*properties.mushy_range_k
    liquidus = properties.melting_temperature_k + 0.5*properties.mushy_range_k
    solid_cp = properties.solid_specific_heat_j_kg_k
    mushy_cp = single_drop.cold_wall_mushy_specific_heat_j_kg_k(properties)
    h_solidus = solid_cp*solidus
    h_liquidus = h_solidus + mushy_cp*properties.mushy_range_k
    result = np.empty_like(enthalpy)
    solid = enthalpy <= h_solidus
    mushy = (~solid) & (enthalpy <= h_liquidus)
    liquid = ~(solid | mushy)
    result[solid] = enthalpy[solid]/solid_cp
    result[mushy] = solidus + (enthalpy[mushy] - h_solidus)/mushy_cp
    if np.any(liquid):
        target = enthalpy[liquid]
        lower = np.full(target.shape, liquidus)
        upper = np.full(target.shape, liquidus + 1000.0)
        cp0 = single_drop.alumina_liquid_specific_heat_j_kg_k(liquidus)
        def liquid_enthalpy(value):
            bounded = np.clip(value, single_drop.ALUMINA_PROPERTY_TEMPERATURE_MIN_K,
                              single_drop.ALUMINA_PROPERTY_TEMPERATURE_MAX_K)
            cp1 = (single_drop.ALUMINA_CP_AT_MELTING_MOLAR_J_MOL_K
                   + single_drop.ALUMINA_CP_MOLAR_SLOPE_J_MOL_K2
                   *(bounded - single_drop.ALUMINA_PROPERTY_TEMPERATURE_MIN_K)
                  )/single_drop.ALUMINA_MOLAR_MASS_KG_MOL
            return h_liquidus + 0.5*(cp0 + cp1)*(value - liquidus)
        for _ in range(40):
            middle = 0.5*(lower + upper)
            below = liquid_enthalpy(middle) < target
            lower[below] = middle[below]
            upper[~below] = middle[~below]
        result[liquid] = 0.5*(lower + upper)
    return result


def coupled_wall_particle_temperature_records():
    select_case()
    _, face_areas = reader.coupled_owner_normals()
    coupled_faces = np.asarray(tuple(face_areas), dtype=np.int64)
    properties = single_drop.read_cold_wall_thermal_properties(COLD)
    records = []
    for directory in reader.numeric_subdirectories(COLD):
        restart = directory/"gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        values = {"bottomLayer": [], "topLayer": [], "enthalpyMean": []}
        for chunk in reader.particle_chunks(restart):
            selected = (chunk["stuck"] != 0) & np.isin(chunk["stuck_face"], coupled_faces)
            if not np.any(selected):
                continue
            enthalpy = chunk["cold_node_specific_enthalpy"][selected]
            active = np.all(np.isfinite(enthalpy) & (enthalpy > 0.0), axis=1)
            if not np.all(active):
                raise RuntimeError(f"Cold-wall particles lack an active enthalpy profile in {restart}")
            values["bottomLayer"].extend(cold_wall_temperature_array(enthalpy[:, 0], properties))
            values["topLayer"].extend(cold_wall_temperature_array(enthalpy[:, -1], properties))
            values["enthalpyMean"].extend(cold_wall_temperature_array(np.mean(enthalpy, axis=1), properties))
        for component, component_values in values.items():
            array = np.asarray(component_values, dtype=float)
            if array.size:
                records.append((float(directory.name), component, int(array.size),
                                float(np.median(array)), float(np.mean(array)),
                                float(np.percentile(array, 25)),
                                float(np.percentile(array, 75))))
    return records, properties


def coupled_wall_particle_temperature_results():
    records, properties = coupled_wall_particle_temperature_records()
    if not records:
        raise RuntimeError("No cold-wall contact-particle temperatures are available")
    with (DATA/f"{PREFIX}_coupled_wall_particle_temperature.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("temperature_component", "time_s", "particle_count",
                         "temperature_median_K", "temperature_mean_K",
                         "temperature_p25_K", "temperature_p75_K"))
        for time, component, count, median, mean, p25, p75 in records:
            writer.writerow((component, f"{time:.17g}", count, f"{median:.12g}",
                             f"{mean:.12g}", f"{p25:.12g}", f"{p75:.12g}"))

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    specifications = (
        ("Bottom layer", "bottomLayer", "#C44E52", "-.", "^"),
        ("Top gas-side layer", "topLayer", "#DD8452", (0, (5, 2)), "D"),
        ("Enthalpy mean", "enthalpyMean", "#55A868", ":", "s"),
    )
    all_times = np.asarray([row[0] for row in records], dtype=float)
    spacing = marker_spacing(all_times[(all_times >= PLOT_START) & (all_times <= PLOT_END)])
    for label, component, color, linestyle, marker in specifications:
        selected = [row for row in records if row[1] == component]
        data = np.asarray([(row[0], row[3], row[5], row[6]) for row in selected])
        data = data[(data[:, 0] >= PLOT_START) & (data[:, 0] <= PLOT_END)]
        axis.fill_between(data[:, 0], data[:, 2], data[:, 3], color=color, alpha=0.12, linewidth=0)
        axis.plot(data[:, 0], data[:, 1], color=color, linestyle=linestyle,
                  linewidth=1.6, marker=marker, markersize=5,
                  markerfacecolor="white", markeredgewidth=0.9,
                  markevery=markevery(data[:, 0], spacing), label=label)
    axis.axhline(properties.melting_temperature_k, color="black", linestyle="--",
                 linewidth=1.2,
                 label=f"Solidification temperature ({properties.melting_temperature_k:.0f} K)")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Wall-contact particle temperature $T_p$ (K)")
    configure_axes(axis)
    inside_legend(axis, "center right")
    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES/f"{PREFIX}_coupled_wall_particle_temperature.png", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    temperature_results()
    heat_flux_results()
    sommerfeld_results()
    radiating_area_results()
    coupled_wall_particle_temperature_results()
    print(f"wrote={RESULTS}")


if __name__ == "__main__":
    main()
