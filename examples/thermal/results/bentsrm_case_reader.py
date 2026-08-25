#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import re
import struct

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 22,
    "legend.fontsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


CASE = Path(__file__).resolve().parent
OUTPUT = CASE.parent / "results"
PREFIX = CASE.name
EXPERIMENT = CASE / "assets/experimental/graphite_thermocouples.csv"
COUPLED_PATCH = "fluid_to_graphite"


def numeric_subdirectories(path):
    result = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                try:
                    result.append((float(child.name), child))
                except ValueError:
                    pass
    return [item[1] for item in sorted(result)]


def _foam_list_body(path):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", path.read_text(errors="replace"), flags=re.S)
    matches = list(re.finditer(r"(?m)^\s*(\d+)\s*\n\s*\(", text))
    if not matches:
        raise RuntimeError(f"Cannot locate OpenFOAM list in {path}")
    match = matches[-1]
    return int(match.group(1)), text[match.end():text.rfind(")")]


def coupled_owner_normals():
    mesh = CASE / "constant/fluid/polyMesh"
    boundary = re.sub(r"/\*.*?\*/|//[^\n]*", " ", (mesh / "boundary").read_text(), flags=re.S)
    patch = re.search(rf"\b{COUPLED_PATCH}\s*\{{(.*?)\}}", boundary, re.S)
    if not patch:
        raise RuntimeError(f"Patch {COUPLED_PATCH} is absent from the fluid mesh")
    start = int(re.search(r"\bstartFace\s+(\d+)\s*;", patch.group(1)).group(1))
    count = int(re.search(r"\bnFaces\s+(\d+)\s*;", patch.group(1)).group(1))
    owner_count, owner_body = _foam_list_body(mesh / "owner")
    owners = np.fromstring(owner_body, sep=" ", dtype=np.int64)
    if owners.size != owner_count:
        raise RuntimeError("Malformed owner list")
    point_count, point_body = _foam_list_body(mesh / "points")
    points = np.asarray([[float(x) for x in item.split()]
                         for item in re.findall(r"\(([^()]*)\)", point_body)], dtype=float)
    if points.shape != (point_count, 3):
        raise RuntimeError("Malformed point list")
    face_count, face_body = _foam_list_body(mesh / "faces")
    faces = [np.fromstring(item, sep=" ", dtype=np.int64)
             for item in re.findall(r"\d+\(([^()]*)\)", face_body)]
    if len(faces) != face_count:
        raise RuntimeError("Malformed face list")
    result = {}
    face_areas = {}
    for face_index in range(start, start + count):
        polygon = points[faces[face_index]]
        area_vector = 0.5*np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
        magnitude = np.linalg.norm(area_vector)
        if not magnitude > 0.0:
            raise RuntimeError(f"Degenerate coupled face {face_index}")
        result.setdefault(int(owners[face_index]), []).append(area_vector/magnitude)
        face_areas[face_index] = float(magnitude)
    return ({cell: np.asarray(normals) for cell, normals in result.items()}, face_areas)


def particle_chunks(path):
    with path.open("rb") as stream:
        header = stream.readline().decode("ascii").split()
        schemas = {f"UGKP_FSH_PARTICLES_SCHEMA{i}_BIN": i for i in range(1, 6)}
        if len(header) != 3 or header[0] not in schemas:
            raise RuntimeError(f"Unsupported particle restart header in {path}")
        schema, total, maximum = schemas[header[0]], int(header[1]), int(header[2])
        emitted = 0
        while emitted < total:
            raw = stream.read(4)
            if len(raw) != 4:
                raise RuntimeError(f"Truncated particle restart {path}")
            count = struct.unpack("<I", raw)[0]
            if count < 1 or count > maximum or emitted + count > total:
                raise RuntimeError(f"Invalid particle chunk in {path}")
            arrays = {}
            for name in ("px", "py", "pz", "ux", "uy", "uz", "T", "theta", "d", "mass"):
                values = np.fromfile(stream, dtype="<f8", count=count)
                if values.size != count:
                    raise RuntimeError(f"Truncated particle restart {path}")
                if name in ("ux", "uy", "uz", "T", "theta", "d", "mass"):
                    arrays[name] = values
            arrays["cell"] = np.fromfile(stream, dtype="<i4", count=count)
            stream.seek(count*4 + count*8*2, 1)
            arrays["stuck"] = np.fromfile(stream, dtype="u1", count=count)
            arrays["stuck_face"] = np.fromfile(stream, dtype="<i4", count=count)
            arrays["contact_area"] = np.fromfile(stream, dtype="<f4", count=count)
            arrays["contact_duration"] = np.zeros(count)
            arrays["contact_maximum_area"] = np.zeros(count)
            arrays["contact_peak_fraction"] = np.zeros(count)
            arrays["cold_frozen_area"] = np.zeros(count)
            arrays["cold_node_specific_enthalpy"] = np.zeros((count, 8))
            if schema >= 2:
                arrays["contact_duration"] = np.fromfile(stream, dtype="<f4", count=count)
                arrays["contact_maximum_area"] = np.fromfile(stream, dtype="<f4", count=count)
            if schema >= 3:
                arrays["contact_peak_fraction"] = np.fromfile(stream, dtype="<f4", count=count)
            if schema >= 4:
                cold_enthalpy = np.fromfile(stream, dtype="<f4", count=count*8)
                if cold_enthalpy.size != count*8:
                    raise RuntimeError(f"Truncated particle restart {path}")
                arrays["cold_node_specific_enthalpy"] = cold_enthalpy.reshape(count, 8)
                stream.seek(count*4*8, 1)
                arrays["cold_frozen_area"] = np.fromfile(stream, dtype="<f4", count=count)
                stream.seek(count*4, 1)
            if schema >= 5:
                stream.seek(count*4*(64 + 8 + 1), 1)
            emitted += count
            yield arrays


def incoming_sommerfeld(path, owner_normals):
    values = []
    speeds = []
    for chunk in particle_chunks(path):
        velocity = np.column_stack((chunk["ux"], chunk["uy"], chunk["uz"]))
        for cell, normals in owner_normals.items():
            indices = np.flatnonzero((chunk["cell"] == cell) & (chunk["stuck"] == 0))
            if indices.size == 0:
                continue
            normal_speed = np.max(velocity[indices] @ normals.T, axis=1)
            incoming = normal_speed > 0.0
            if not np.any(incoming):
                continue
            indices = indices[incoming]
            normal_speed = normal_speed[incoming]
            temperature = np.clip(chunk["T"][indices], 2327.0, 4000.0)
            diameter = chunk["d"][indices]
            density = 1000.0*(2.917 - 1.228e-4*(temperature - 2327.0))
            surface_tension = 0.632 - 2.310e-5*(temperature - 2327.0)
            viscosity = 5.77e-4*np.exp(9743.0/temperature)
            weber = density*normal_speed**2*diameter/surface_tension
            reynolds = density*normal_speed*diameter/viscosity
            valid = (diameter > 0.0) & np.isfinite(weber) & np.isfinite(reynolds)
            values.extend((np.sqrt(weber[valid])*reynolds[valid]**0.25).tolist())
            speeds.extend(normal_speed[valid].tolist())
    return np.asarray(values), np.asarray(speeds)


def normalized_radiating_area(path, face_areas):
    props = (CASE / "constant/particleProperties").read_text(errors="replace")
    rho_s = float(re.search(r"(?m)^\s*rhoS\s+([-+0-9.eE]+)\s*;", props).group(1))
    coverage = float(re.search(r"(?m)^\s*maximumCoverage\s+([-+0-9.eE]+)\s*;", props).group(1))
    represented = {face: 0.0 for face in face_areas}
    for chunk in particle_chunks(path):
        for face in face_areas:
            indices = np.flatnonzero((chunk["stuck"] != 0) & (chunk["stuck_face"] == face))
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
                normalized_age = np.divide(age, duration, out=np.zeros_like(age), where=duration > 0.0)
                peak = chunk["contact_peak_fraction"][ti]
                kinematic_fraction = np.zeros_like(age)
                rising = normalized_age <= peak
                x = np.divide(normalized_age, peak, out=np.zeros_like(age), where=peak > 0.0)
                kinematic_fraction[rising] = (x*(2.0 - x))[rising]
                falling = ~rising
                y = np.divide(normalized_age - peak, 1.0 - peak,
                              out=np.zeros_like(age), where=peak < 1.0)
                kinematic_fraction[falling] = np.cos(0.5*np.pi*y[falling])**2
                kinematic_fraction[(normalized_age <= 0.0) | (normalized_age >= 1.0)] = 0.0
                kinematic = chunk["contact_maximum_area"][ti]*kinematic_fraction
                physical[transient] = np.maximum(
                    np.maximum(kinematic, chunk["cold_frozen_area"][ti])
                    - chunk["contact_area"][ti], 0.0)
            diameter = chunk["d"][indices]
            physical_mass = (np.pi/6.0)*rho_s*diameter**3
            valid = (physical > 0.0) & (physical_mass > 0.0)
            represented[face] += float(np.sum(
                physical[valid]*chunk["mass"][indices[valid]]/physical_mass[valid]))
    total_area = sum(face_areas.values())
    occupied_area = sum(min(represented[face], coverage*face_areas[face]) for face in face_areas)
    return 1.0 - occupied_area/total_area, occupied_area/total_area, occupied_area, total_area


def read_probe_history():
    rows = {}
    root = CASE / "postProcessing/graphiteDepthTemperature/graphite"
    for directory in numeric_subdirectories(root):
        source = directory / "T"
        if not source.is_file():
            continue
        for line in source.read_text(errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            if len(values) >= 5:
                rows[float(values[0])] = tuple(float(value) for value in values[1:5])
    if not rows:
        raise RuntimeError(f"No graphite probe data found below {root}")
    times = np.array(sorted(rows), dtype=float)
    temperatures = np.array([rows[time] for time in times], dtype=float)
    return times, temperatures


def _graphite_depth_cell_centres():
    source = CASE / "system/graphite/blockMeshDict"
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source.read_text(errors="replace"), flags=re.S)
    vertices_match = re.search(
        r"\bvertices\s*\(\s*((?:\([^()]+\)\s*)+)\)\s*;", text, re.S
    )
    block_match = re.search(
        r"\bhex\s*\(([^()]*)\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text
    )
    if not vertices_match or not block_match:
        raise RuntimeError(f"Cannot recover graphite block geometry from {source}")
    vertices = np.asarray(
        [[float(value) for value in item.split()]
         for item in re.findall(r"\(([^()]*)\)", vertices_match.group(1))],
        dtype=float,
    )
    block_vertices = np.fromstring(block_match.group(1), sep=" ", dtype=np.int64)
    nx, ny, nz = (int(block_match.group(index)) for index in range(2, 5))
    if block_vertices.size != 8 or vertices.shape[1] != 3 or min(nx, ny, nz) < 1:
        raise RuntimeError(f"Unsupported graphite block geometry in {source}")
    interface_centre = np.mean(vertices[block_vertices[:4]], axis=0)
    back_centre = np.mean(vertices[block_vertices[4:]], axis=0)
    thickness = float(np.linalg.norm(back_centre - interface_centre))
    if not thickness > 0.0:
        raise RuntimeError(f"Non-positive graphite thickness in {source}")
    depth_centres = (np.arange(nz, dtype=float) + 0.5)*thickness/nz
    return depth_centres, nx*ny


def _read_internal_scalar_field(path, expected_count):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", path.read_text(errors="replace"), flags=re.S)
    uniform = re.search(r"\binternalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform:
        return np.full(expected_count, float(uniform.group(1)), dtype=float)
    nonuniform = re.search(
        r"\binternalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if not nonuniform:
        raise RuntimeError(f"Cannot read graphite internal temperature field {path}")
    declared = int(nonuniform.group(1))
    values = np.fromstring(nonuniform.group(2), sep=" ", dtype=float)
    if declared != expected_count or values.size != expected_count or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Malformed graphite internal temperature field {path}")
    return values


def read_graphite_depth_band_history(depths_mm=(5.0, 10.0, 15.0, 20.0), half_width_mm=2.0):
    depth_centres, cells_per_depth = _graphite_depth_cell_centres()
    selections = []
    for depth_mm in depths_mm:
        selected = np.flatnonzero(
            np.abs(depth_centres - depth_mm*1.0e-3)
            <= half_width_mm*1.0e-3 + 1.0e-12
        )
        if selected.size == 0:
            raise RuntimeError(
                f"No graphite cells lie within {depth_mm:g} +/- {half_width_mm:g} mm"
            )
        selections.append(selected)
    rows = []
    for directory in numeric_subdirectories(CASE):
        source = directory / "graphite/T"
        if not source.is_file():
            continue
        temperature = _read_internal_scalar_field(
            source, depth_centres.size*cells_per_depth
        )
        centreline_profile = np.mean(
            temperature.reshape(depth_centres.size, cells_per_depth), axis=1
        )
        means = np.asarray([np.mean(centreline_profile[selected]) for selected in selections])
        minima = np.asarray([np.min(centreline_profile[selected]) for selected in selections])
        maxima = np.asarray([np.max(centreline_profile[selected]) for selected in selections])
        rows.append((float(directory.name), means, minima, maxima))
    if not rows:
        raise RuntimeError(f"No graphite temperature fields found in written times below {CASE}")
    return (
        np.asarray([row[0] for row in rows], dtype=float),
        np.asarray([row[1] for row in rows], dtype=float),
        np.asarray([row[2] for row in rows], dtype=float),
        np.asarray([row[3] for row in rows], dtype=float),
    )


def read_experiment():
    data = np.genfromtxt(EXPERIMENT, delimiter=",", names=True, encoding="utf-8")
    return np.asarray(data["time_s"], dtype=float), np.column_stack(
        [np.asarray(data[f"TC{i}_{depth}mm_K"], dtype=float) for i, depth in enumerate((5, 10, 15, 20), 1)]
    )


def read_wall_heat_flux_history():
    rows = {}
    root = CASE / "postProcessing/fluid/coupledWallHeatFluxes"
    for directory in numeric_subdirectories(root):
        source = directory / "surfaceFieldValue.dat"
        if not source.is_file():
            continue
        columns = None
        for line in source.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header = stripped.lstrip("#").split()
                if header and header[0].lower() == "time":
                    columns = [item[item.find("(") + 1:item.rfind(")")] if "(" in item else item
                               for item in header]
                continue
            values = [float(value) for value in stripped.split()]
            if columns is None or len(columns) != len(values):
                continue
            record = dict(zip(columns, values))
            required = (
                "particleRadiationWallHeatFlux",
                "particleReflectedWallHeatFlux",
                "particleStuckWallHeatFlux",
                "gasConvectiveWallHeatFlux",
            )
            if not all(name in record for name in required):
                continue
            rows[record["Time"]] = tuple(record[name] for name in required)
    if not rows:
        raise RuntimeError(f"No complete four-component coupled-wall area-average data found below {root}")
    times = np.array(sorted(rows), dtype=float)
    fluxes = np.array([rows[time] for time in times], dtype=float)
    return times, fluxes


def configure_axes(axis):
    axis.grid(False)
    axis.tick_params(direction="in", top=True, right=True, length=6, width=1.0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)


def write_temperature_results():
    simulation_time, calculated = read_probe_history()
    experiment_time, measured = read_experiment()
    measured_at_simulation_time = np.column_stack(
        [np.interp(simulation_time, experiment_time, measured[:, index]) for index in range(4)]
    )
    destination = OUTPUT / f"{PREFIX}_temperature_comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["time_s"]
        for depth in (5, 10, 15, 20):
            header.extend((f"calculated_{depth}mm_K", f"measured_{depth}mm_K"))
        writer.writerow(header)
        for row_index, time in enumerate(simulation_time):
            row = [f"{time:.17g}"]
            for column in range(4):
                row.extend((f"{calculated[row_index, column]:.12g}", f"{measured_at_simulation_time[row_index, column]:.12g}"))
            writer.writerow(row)

    plot_start = 1.6
    plot_end = 2.6
    simulation_plot = (simulation_time >= plot_start) & (simulation_time <= plot_end)
    experiment_plot = (experiment_time >= plot_start) & (experiment_time <= plot_end)
    colors = ("#C44E52", "#4C72B0", "#55A868", "#DD8452")
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    for index, depth in enumerate((5, 10, 15, 20)):
        axis.plot(experiment_time[experiment_plot], measured[experiment_plot, index], color=colors[index],
                  linestyle="--", linewidth=1.2, marker="o", markersize=5, markerfacecolor="white",
                  markeredgewidth=0.9, label=f"Measured {depth} mm")
        axis.plot(simulation_time[simulation_plot], calculated[simulation_plot, index], color=colors[index],
                  linestyle="-", linewidth=1.6, label=f"Calculated {depth} mm")
    axis.set_xlim(plot_start, plot_end)
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Temperature $T$ (K)")
    axis.legend(loc="upper left", frameon=False, ncol=2, fontsize=14,
                handlelength=1.5, columnspacing=0.8, labelspacing=0.3)
    configure_axes(axis)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{PREFIX}_temperature_comparison.png", dpi=600, facecolor="white")
    plt.close(fig)


def write_heat_flux_results():
    times, fluxes = read_wall_heat_flux_history()
    radiation, reflected, deposited, convection = fluxes.T
    total = radiation + reflected + deposited + convection
    destination = OUTPUT / f"{PREFIX}_wall_heat_flux.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "total_wall_heat_flux_W_m2", "radiation_heat_flux_W_m2",
                         "reflected_particle_heat_flux_W_m2", "deposited_particle_heat_flux_W_m2",
                         "gas_convective_heat_flux_W_m2"))
        for values in zip(times, total, radiation, reflected, deposited, convection):
            writer.writerow([f"{value:.17g}" for value in values])

    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    axis.plot(times, total, color="#C44E52", linewidth=2.2, marker="o", markersize=5,
              markerfacecolor="white", label="Total heat flux")
    axis.plot(times, radiation, color="#4C72B0", linewidth=1.8, linestyle="--", label="Radiation")
    axis.plot(times, reflected, color="#55A868", linewidth=1.8, linestyle="-.", label="Reflected particles")
    axis.plot(times, deposited, color="#DD8452", linewidth=1.8, linestyle=":", label="Deposited particles")
    axis.plot(times, convection, color="#6B7A8F", linewidth=1.8, linestyle=(0, (5, 2)), label="Gas convection")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Wall heat flux $q''$ (W m$^{-2}$)")
    axis.legend(frameon=False)
    configure_axes(axis)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{PREFIX}_wall_heat_flux.png", dpi=600, facecolor="white")
    plt.close(fig)


def write_sommerfeld_results():
    owner_normals, _ = coupled_owner_normals()
    records = []
    for directory in numeric_subdirectories(CASE):
        restart = directory / "gpuResidentStrictParticles.dat"
        if not restart.is_file():
            continue
        sommerfeld, normal_speed = incoming_sommerfeld(restart, owner_normals)
        if sommerfeld.size:
            records.append((float(directory.name), int(sommerfeld.size),
                            float(np.median(sommerfeld)), float(np.mean(sommerfeld)),
                            float(np.percentile(sommerfeld, 25)), float(np.percentile(sommerfeld, 75)),
                            float(np.median(normal_speed))))
        else:
            records.append((float(directory.name), 0, math.nan, math.nan,
                            math.nan, math.nan, math.nan))
    if not records:
        raise RuntimeError("No particle restart write times found for Sommerfeld statistics")
    destination = OUTPUT / f"{PREFIX}_coupled_wall_incoming_sommerfeld.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "incoming_particle_count", "sommerfeld_median",
                         "sommerfeld_mean", "sommerfeld_p25", "sommerfeld_p75",
                         "normal_speed_median_m_s"))
        for record in records:
            writer.writerow([record[1] if index == 1 else f"{value:.17g}"
                             for index, value in enumerate(record)])
    data = np.asarray(records, dtype=float)
    valid = np.isfinite(data[:, 2])
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    axis.fill_between(data[valid, 0], data[valid, 4], data[valid, 5],
                      color="#4C72B0", alpha=0.18, linewidth=0, label="25%--75% interval")
    axis.plot(data[valid, 0], data[valid, 2], color="#C44E52", linewidth=1.8,
              marker="o", markersize=5, markerfacecolor="white", markeredgewidth=0.9,
              label="Median Sommerfeld number")
    axis.axhline(20.0, color="black", linestyle="--", linewidth=1.5,
                 label="Deposition threshold")
    axis.set_yscale("log")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Incoming-particle Sommerfeld number $K$")
    axis.legend(frameon=False)
    configure_axes(axis)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{PREFIX}_coupled_wall_incoming_sommerfeld.png",
                dpi=600, facecolor="white")
    plt.close(fig)


def write_radiating_area_results():
    _, face_areas = coupled_owner_normals()
    records = []
    for directory in numeric_subdirectories(CASE):
        restart = directory / "gpuResidentStrictParticles.dat"
        if restart.is_file():
            records.append((float(directory.name),) + normalized_radiating_area(restart, face_areas))
    if not records:
        raise RuntimeError("No particle restart write times found for radiating-area statistics")
    destination = OUTPUT / f"{PREFIX}_coupled_wall_radiating_area_fraction.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "radiating_area_fraction", "contact_area_fraction",
                         "effective_contact_area_m2", "coupled_wall_area_m2"))
        writer.writerows([[f"{value:.17g}" for value in record] for record in records])
    data = np.asarray(records)
    fig, axis = plt.subplots(figsize=(10, 8), facecolor="white")
    axis.plot(data[:, 0], data[:, 1], color="#C44E52", linewidth=1.8,
              marker="o", markersize=5, markerfacecolor="white", markeredgewidth=0.9,
              label="Radiating area fraction")
    axis.set_xlabel("Time $t$ (s)")
    axis.set_ylabel("Normalized radiating area $1-A_c/A_w$")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False)
    configure_axes(axis)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{PREFIX}_coupled_wall_radiating_area_fraction.png",
                dpi=600, facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    write_temperature_results()
    write_heat_flux_results()
    write_sommerfeld_results()
    write_radiating_area_results()
    print(f"wrote={OUTPUT}")
