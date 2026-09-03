#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[2]
THERMAL = CASE_ROOT.parent
OUTPUT = THERMAL / "results" / CASE_ROOT.name
DATA = OUTPUT / "data"
FIGURES = OUTPUT / "figures"
PROJECT = THERMAL.parents[1]
sys.path.insert(0, str(PROJECT / "tools/postprocessing"))
sys.path.insert(0, str(THERMAL / "postprocessing"))
from bartz import bartz_wall_heat_flux
import bentsrm_case_reader as particle_reader


plt.rcParams.update({
    "font.family": "serif", "font.size": 17, "axes.labelsize": 20,
    "legend.fontsize": 14, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix", "axes.unicode_minus": False,
})


def numeric_directories(path: Path):
    result = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                try:
                    result.append((float(child.name), child))
                except ValueError:
                    pass
    return sorted(result)


def clean_text(path: Path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", path.read_text(errors="replace"), flags=re.S)


def patch_values(path: Path, patch: str, expected: int):
    text = clean_text(path)
    match = re.search(rf"\b{re.escape(patch)}\s*\{{(.*?)\}}", text, re.S)
    if not match:
        raise RuntimeError(f"patch {patch} missing in {path}")
    body = match.group(1)
    uniform = re.search(r"\bvalue\s+uniform\s+([-+0-9.eE]+)\s*;", body)
    if uniform:
        return np.full(expected, float(uniform.group(1)))
    nonuniform = re.search(r"\bvalue\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", body, re.S)
    if not nonuniform:
        raise RuntimeError(f"patch values missing in {path}")
    values = np.fromstring(nonuniform.group(2), sep=" ")
    if int(nonuniform.group(1)) != expected or values.size != expected:
        raise RuntimeError(f"patch size mismatch in {path}")
    return values


def pressure_table(path: Path):
    pairs = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", path.read_text(errors="replace"))
    values = np.asarray([(float(a), float(b)) for a, b in pairs])
    return values[:, 0], values[:, 1]


def run_probes(case: Path):
    command = ["postProcess", "-region", "graphite", "-dict", "system/mss7TemperatureProbesDict", "-time", "1:"]
    completed = subprocess.run(command, cwd=case, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("temperature probe extraction failed:\n" + completed.stdout[-4000:])


def read_probe_series(path: Path):
    rows = {}
    coordinates = []
    for _, directory in numeric_directories(path):
        source = directory / "T"
        if not source.is_file():
            continue
        local = []
        for line in source.read_text(errors="replace").splitlines():
            match = re.match(r"#\s*Probe\s+\d+\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", line)
            if match:
                local.append(tuple(float(value) for value in match.groups()))
            elif line.strip() and not line.lstrip().startswith("#"):
                values = [float(value) for value in line.split()]
                rows[values[0]] = values[1:]
        if local and not coordinates:
            coordinates = local
    if not rows:
        raise RuntimeError(f"no probe data below {path}")
    times = np.asarray(sorted(rows))
    return times, np.asarray([rows[time] for time in times]), coordinates


def wall_temperature(times, values, coordinates, target_x=0.151945102963):
    x = np.asarray([point[0] for point in coordinates])
    order = np.argsort(x)
    return times, np.asarray([np.interp(target_x, x[order], row[order]) for row in values])


def matched_temperature(case: Path, baseline: Path):
    run_probes(case)
    tw_t, tw_v, tw_c = read_probe_series(case / "postProcessing/mss7ThroatWallProbe/graphite")
    tw_t, tw = wall_temperature(tw_t, tw_v, tw_c)
    ti_t, ti, _ = read_probe_series(case / "postProcessing/mss7ThroatTemperatureProbes/graphite")
    bt_t, bt_v, bt_c = read_probe_series(baseline / "postProcessing/mss7ThroatWallProbe/graphite")
    bt_t, btw = wall_temperature(bt_t, bt_v, bt_c)
    bi_t, bi, _ = read_probe_series(baseline / "postProcessing/mss7ThroatTemperatureProbes/graphite")
    lower = max(float(np.min(tw_t)), float(np.min(ti_t)), float(np.min(bt_t)), float(np.min(bi_t)))
    upper = min(float(np.max(tw_t)), float(np.max(ti_t)), float(np.max(bt_t)), float(np.max(bi_t)))
    two_common = sorted(set(np.round(tw_t, 8)) & set(np.round(ti_t, 8)))
    two_times = np.asarray([value for value in two_common if lower - 1.0e-8 <= value <= upper + 1.0e-8])
    if two_times.size == 0:
        raise RuntimeError("two-phase and pure-gas probe ranges do not overlap")
    two_phase = np.column_stack((np.interp(two_times, tw_t, tw), np.interp(two_times, ti_t, ti[:, 0]), np.interp(two_times, ti_t, ti[:, 1])))
    pure_gas = np.column_stack((np.interp(two_times, bt_t, btw), np.interp(two_times, bi_t, bi[:, 0]), np.interp(two_times, bi_t, bi[:, 1])))
    return two_times, two_phase, pure_gas


def foam_list(path: Path):
    text = clean_text(path)
    matches = list(re.finditer(r"(?m)^\s*(\d+)\s*\n\s*\(", text))
    if not matches:
        raise RuntimeError(f"OpenFOAM list missing in {path}")
    match = matches[-1]
    return int(match.group(1)), text[match.end():text.rfind(")")]


def coupled_face_geometry(case: Path, patch_name: str):
    mesh = case / "constant/fluid/polyMesh"
    boundary = clean_text(mesh / "boundary")
    patch = re.search(rf"\b{re.escape(patch_name)}\s*\{{(.*?)\}}", boundary, re.S)
    if not patch:
        raise RuntimeError(f"patch {patch_name} is absent from the fluid mesh")
    start = int(re.search(r"\bstartFace\s+(\d+)\s*;", patch.group(1)).group(1))
    count = int(re.search(r"\bnFaces\s+(\d+)\s*;", patch.group(1)).group(1))
    point_count, point_body = foam_list(mesh / "points")
    points = np.asarray([[float(x) for x in item.split()] for item in re.findall(r"\(([^()]*)\)", point_body)])
    face_count, face_body = foam_list(mesh / "faces")
    faces = [np.fromstring(item, sep=" ", dtype=np.int64) for item in re.findall(r"\d+\(([^()]*)\)", face_body)]
    if points.shape != (point_count, 3) or len(faces) != face_count:
        raise RuntimeError("malformed fluid mesh")
    face_ids = np.arange(start, start + count, dtype=np.int64)
    centres = np.empty((count, 3))
    areas = np.empty(count)
    for local, face_id in enumerate(face_ids):
        polygon = points[faces[int(face_id)]]
        centres[local] = np.mean(polygon, axis=0)
        area_vector = 0.5*np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
        areas[local] = np.linalg.norm(area_vector)
    return face_ids, centres, areas


def completed_radiation_time(path: Path):
    state = path / "thermalExchangeState"
    manifest = path / "thermalExchangeManifest"
    if not state.is_file() or not manifest.is_file():
        return None
    match = re.search(r"\bcompletedRadiationSimulationTimeS\s+([-+0-9.eE]+)\s*;", state.read_text(errors="replace"))
    return None if not match else float(match.group(1))


def radiation_directories(case: Path):
    result = []
    previous = 1.0
    for _, directory in numeric_directories(case):
        time_s = completed_radiation_time(directory)
        if time_s is not None and time_s > previous + 1.0e-10:
            result.append((time_s, directory))
            previous = time_s
    return result


def effective_radiating_area(path: Path, face_ids, face_areas, properties_path: Path):
    props = properties_path.read_text(errors="replace")
    rho_s = float(re.search(r"(?m)^\s*rhoS\s+([-+0-9.eE]+)\s*;", props).group(1))
    coverage = float(re.search(r"(?m)^\s*maximumCoverage\s+([-+0-9.eE]+)\s*;", props).group(1))
    represented = np.zeros(face_ids.size)
    first, last = int(face_ids[0]), int(face_ids[-1])
    for chunk in particle_reader.particle_chunks(path):
        indices = np.flatnonzero((chunk["stuck"] != 0) & (chunk["stuck_face"] >= first) & (chunk["stuck_face"] <= last))
        if indices.size == 0:
            continue
        state = chunk["stuck"][indices]
        physical = np.zeros(indices.size)
        deposited = state == 1
        physical[deposited] = chunk["contact_area"][indices[deposited]]
        transient = (state == 2) | (state == 3)
        if np.any(transient):
            ti = indices[transient]
            duration = chunk["contact_duration"][ti]
            age = np.clip(chunk["theta"][ti], 0.0, duration)
            normalized = np.divide(age, duration, out=np.zeros_like(age), where=duration > 0.0)
            peak = chunk["contact_peak_fraction"][ti]
            fraction = np.zeros_like(age)
            rising = normalized <= peak
            x = np.divide(normalized, peak, out=np.zeros_like(age), where=peak > 0.0)
            fraction[rising] = (x*(2.0 - x))[rising]
            falling = ~rising
            y = np.divide(normalized - peak, 1.0 - peak, out=np.zeros_like(age), where=peak < 1.0)
            fraction[falling] = np.cos(0.5*np.pi*y[falling])**2
            fraction[(normalized <= 0.0) | (normalized >= 1.0)] = 0.0
            kinematic = chunk["contact_maximum_area"][ti]*fraction
            physical[transient] = np.maximum(np.maximum(kinematic, chunk["cold_frozen_area"][ti]) - chunk["contact_area"][ti], 0.0)
        diameter = chunk["d"][indices]
        physical_mass = (np.pi/6.0)*rho_s*diameter**3
        valid = (physical > 0.0) & (physical_mass > 0.0)
        local_face = chunk["stuck_face"][indices[valid]].astype(np.int64) - first
        contribution = physical[valid]*chunk["mass"][indices[valid]]/physical_mass[valid]
        np.add.at(represented, local_face, contribution)
    contact = np.minimum(represented, coverage*face_areas)
    effective = face_areas - contact
    return effective, effective/face_areas, contact


def write_rows(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def time_name(value: float):
    return f"{value:.9f}".rstrip("0").rstrip(".")


def temperature_figure(case: Path, baseline: Path):
    two_times, two_phase, pure_gas = matched_temperature(case, baseline)
    rows = []
    labels = ("wall", "10mm", "30mm")
    for index, time_s in enumerate(two_times):
        row = {"time_s": time_s}
        for column, label in enumerate(labels):
            row[f"{label}_two_phase_K"] = two_phase[index, column]
            row[f"{label}_pure_gas_K"] = pure_gas[index, column]
        rows.append(row)
    write_rows(DATA / "temperature_comparison.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    for column, (axis, label) in enumerate(zip(axes, ("Wall", "10 mm", "30 mm"))):
        axis.plot(two_times, pure_gas[:, column], color="#1f77b4", lw=2.2, label="Pure gas")
        axis.plot(two_times, two_phase[:, column], color="#d62728", lw=2.0, label="Two phase")
        axis.set_xlim(float(two_times[0]), float(two_times[-1]))
        axis.set_xlabel(r"Time $t$ (s)")
        axis.set_title(label)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"Temperature $T$ (K)")
    axes[0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "temperature_comparison.png", dpi=300, facecolor="white")
    plt.close(fig)


def spatial_figures(case: Path, patch: str):
    face_ids, centres, face_areas = coupled_face_geometry(case, patch)
    coordinate_mm = centres[:, 0]*1000.0
    radius = np.sqrt(centres[:, 1]**2 + centres[:, 2]**2)
    throat_x = float(centres[np.argmin(radius), 0])
    order = np.argsort(coordinate_mm)
    config = json.loads((case / "assets/postprocessing/bartz.json").read_text())
    pressure_time, pressure = pressure_table(case / "constant/fluid/inletPressure.table")
    heat_data = DATA / "wall_heat_flux_profiles"
    area_data = DATA / "effective_radiating_area_profiles"
    heat_figures = FIGURES / "wall_heat_flux_profiles"
    area_figures = FIGURES / "effective_radiating_area_profiles"
    for directory in (heat_data, area_data, heat_figures, area_figures):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    fields = {
        "Radiation": "particleRadiationWallHeatFlux",
        "Reflection": "particleReflectedWallHeatFlux",
        "Deposition": "particleStuckWallHeatFlux",
        "Gas convection": "gasConvectiveWallHeatFlux",
    }
    records = radiation_directories(case)
    for time_s, directory in records:
        values = {label: patch_values(directory / "fluid" / field, patch, face_ids.size) for label, field in fields.items()}
        wall_temperature = patch_values(directory / "fluid/T", patch, face_ids.size)
        chamber_pressure = float(np.interp(time_s, pressure_time, pressure))
        bartz_result = [
            bartz_wall_heat_flux(
                chamber_pressure, float(wall_temperature[i]), float(radius[i]), float(centres[i, 0]), throat_x,
                config["total_temperature_k"], config["throat_diameter_m"], config["throat_curvature_radius_m"],
                config["dynamic_viscosity_pa_s"], config["specific_heat_j_kg_k"], config["prandtl"],
                config["gamma"], config["gas_constant_j_kg_k"],
            ) for i in range(face_ids.size)
        ]
        bartz = np.asarray([item[0] for item in bartz_result])
        bartz_h = np.asarray([item[1] for item in bartz_result])
        bartz_mach = np.asarray([item[2] for item in bartz_result])
        effective, fraction, contact = effective_radiating_area(
            directory / "gpuResidentStrictParticles.dat", face_ids, face_areas, case / "constant/particleProperties"
        )
        name = time_name(time_s)
        heat_rows, area_rows = [], []
        for i in order:
            heat_rows.append({
                "time_s": time_s, "wall_coordinate_mm": coordinate_mm[i],
                "radiation_W_m2": values["Radiation"][i], "reflection_W_m2": values["Reflection"][i],
                "deposition_W_m2": values["Deposition"][i], "convection_W_m2": values["Gas convection"][i],
                "bartz_W_m2": bartz[i], "bartz_h_W_m2_K": bartz_h[i], "bartz_mach": bartz_mach[i],
                "local_radius_m": radius[i], "wall_temperature_K": wall_temperature[i],
            })
            area_rows.append({
                "time_s": time_s, "wall_coordinate_mm": coordinate_mm[i],
                "effective_radiating_area_m2": effective[i], "effective_radiating_area_fraction": fraction[i],
                "contact_area_m2": contact[i], "face_area_m2": face_areas[i],
            })
        write_rows(heat_data / f"{name}.csv", heat_rows)
        write_rows(area_data / f"{name}.csv", area_rows)
        fig, axis = plt.subplots(figsize=(9.4, 6.0))
        styles = (("Radiation", "#9467bd"), ("Reflection", "#d62728"), ("Deposition", "#1f77b4"),
                  ("Gas convection", "#2ca02c"), ("Bartz", "#ff7f0e"))
        for label, color in styles:
            values_to_plot = bartz if label == "Bartz" else values[label]
            axis.plot(coordinate_mm[order], values_to_plot[order]/1.0e6, lw=2.0, color=color, label=label)
        axis.set_xlabel("Axial wall coordinate (mm)")
        axis.set_ylabel(r"Wall heat flux (MW m$^{-2}$)")
        axis.set_title(rf"$t={time_s:.6f}$ s")
        axis.legend(loc="best", frameon=False, ncol=2)
        axis.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(heat_figures / f"{name}.png", dpi=300, facecolor="white")
        fig.savefig(FIGURES / "wall_heat_flux_profile.png", dpi=300, facecolor="white")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(9.4, 6.0))
        axis.plot(coordinate_mm[order], fraction[order], color="#1f77b4", lw=2.2)
        axis.set_xlabel("Axial wall coordinate (mm)")
        axis.set_ylabel(r"Normalized radiating area $A_{\mathrm{rad}}/A_f$")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(rf"$t={time_s:.6f}$ s")
        axis.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(area_figures / f"{name}.png", dpi=300, facecolor="white")
        fig.savefig(FIGURES / "effective_radiating_area_profile.png", dpi=300, facecolor="white")
        plt.close(fig)
        shutil.copy2(heat_data / f"{name}.csv", DATA / "wall_heat_flux_profile.csv")
        shutil.copy2(area_data / f"{name}.csv", DATA / "effective_radiating_area_profile.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    particle_reader.CASE = case
    particle_reader.COUPLED_PATCH = "fluid_to_graphite"
    temperature_figure(case, case.parent / "MSS7_turbulent_wallModel")
    spatial_figures(case, "fluid_to_graphite")
    print(f"data={DATA}")
    print(f"figures={FIGURES}")


if __name__ == "__main__":
    main()
