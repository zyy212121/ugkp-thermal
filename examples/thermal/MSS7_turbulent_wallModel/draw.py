#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_CASE = Path(__file__).resolve().parent
THERMAL = SCRIPT_CASE.parent
LAMINAR_CASE = THERMAL / "MSS7_laminar"
SST_CASE = THERMAL / "MSS7_turbulent_wallModel"
BENCHMARK_CASE = THERMAL / "MSS7_turbulent_wallModel_benchmark"
SHARED_ASSETS = SST_CASE / "assets/postprocessing"
ASSET_DATA = SHARED_ASSETS / "data"
COMPARISON_RESULTS = THERMAL / "results/MSS7_turbulent_wallModel"
EXPERIMENT = ASSET_DATA / "mss7_temperature_histories_digitized.csv"
PROBE_LOCATIONS = ASSET_DATA / "mss7_temperature_probe_locations.csv"
TIME_TOLERANCE_S = 1.0e-8

sys.path.insert(0, str(SHARED_ASSETS))
from bartz import bartz_wall_heat_flux


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


def numeric_directories(path: Path) -> list[tuple[float, Path]]:
    directories: list[tuple[float, Path]] = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                try:
                    directories.append((float(child.name), child))
                except ValueError:
                    pass
    return sorted(directories)


def clean_text(path: Path) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", path.read_text(errors="replace"), flags=re.S)


def foam_list(path: Path) -> tuple[int, str]:
    text = clean_text(path)
    matches = list(re.finditer(r"(?m)^\s*(\d+)\s*\n\s*\(", text))
    if not matches:
        raise RuntimeError(f"OpenFOAM list missing in {path}")
    match = matches[-1]
    return int(match.group(1)), text[match.end():text.rfind(")")]


def coupled_face_geometry(case: Path, patch_name: str, region: str = "fluid", axisymmetric: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mesh = case / "constant" / region / "polyMesh"
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
    centres = np.empty((count, 3))
    for local, face_id in enumerate(range(start, start + count)):
        polygon = points[faces[face_id]]
        centres[local] = np.mean(polygon, axis=0)
        if axisymmetric:
            radius = np.mean(np.linalg.norm(polygon[:, 1:], axis=1))
            centres[local, 1:] *= radius/np.linalg.norm(centres[local, 1:])
    return np.arange(start, start + count, dtype=np.int64), centres


def patch_values(path: Path, patch: str, expected: int) -> np.ndarray:
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


def interval_wall_temperature(case: Path, directory: Path, patch: str, expected: int) -> tuple[np.ndarray, str]:
    state = clean_text(directory / "thermalExchangeState")
    if not re.search(r"\binitialState\s+true\s*;", state):
        return patch_values(directory / "fluid/T", patch, expected), "previous_fluid_boundary"
    config = clean_text(case / "constant/solidRegionProperties")
    if not re.search(r"\bmapping\s+oneToOneConformal\s*;", config):
        raise RuntimeError("initial wall-temperature reconstruction requires oneToOneConformal mapping")
    region_match = re.search(r"\bsolidRegion\s+(\w+)\s*;", config)
    pairs = re.findall(r"\{[^{}]*\bfluidPatch\s+" + re.escape(patch) + r"\s*;[^{}]*\bsolidPatch\s+(\w+)\s*;[^{}]*\}", config)
    if region_match is None or len(pairs) != 1:
        raise RuntimeError(f"initial solid coupling pair is ambiguous for {patch}")
    region, solid_patch = region_match.group(1), pairs[0]
    fluid_ids, fluid_centres = coupled_face_geometry(case, patch)
    solid_ids, solid_centres = coupled_face_geometry(case, solid_patch, region)
    if len(fluid_ids) != expected or len(solid_ids) != expected or not np.allclose(fluid_centres, solid_centres, rtol=0, atol=1e-9):
        raise RuntimeError("initial solid/fluid coupling faces do not match in solver order")
    owner_count, owner_body = foam_list(case / "constant" / region / "polyMesh/owner")
    owners = np.fromstring(owner_body, sep=" ", dtype=np.int64)
    if owners.size != owner_count or solid_ids[-1] >= owner_count:
        raise RuntimeError("malformed solid face owner list")
    text = clean_text(directory / region / "T")
    uniform = re.search(r"\binternalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform:
        values = np.full(expected, float(uniform.group(1)))
    else:
        field = re.search(r"\binternalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
        if field is None:
            raise RuntimeError(f"solid internal temperature missing in {directory / region / 'T'}")
        internal = np.fromstring(field.group(2), sep=" ")
        selected = owners[solid_ids]
        if internal.size != int(field.group(1)) or np.any(selected < 0) or np.any(selected >= internal.size):
            raise RuntimeError("invalid solid temperature/owner addressing")
        values = internal[selected]
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise RuntimeError("invalid initial coupled wall temperature")
    return values, "initial_solid_owner_mapped"


def pressure_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pairs = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", path.read_text(errors="replace"))
    values = np.asarray([(float(a), float(b)) for a, b in pairs])
    if values.shape[0] == 0:
        raise RuntimeError(f"pressure schedule missing in {path}")
    return values[:, 0], values[:, 1]


def foam_file_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_text(errors="replace").rstrip().endswith("// ************************************************************************* //")


def completed_exchange_time(path: Path) -> float:
    text = clean_text(path)
    match = re.search(r"\bcompletedSimulationTimeS\s+([-+0-9.eE]+)\s*;", text)
    if not match:
        raise RuntimeError(f"completedSimulationTimeS missing in {path}")
    return float(match.group(1))


def previous_exchange_time(path: Path) -> float:
    text = clean_text(path)
    match = re.search(r"\bpreviousExchangeSimulationTimeS\s+([-+0-9.eE]+)\s*;", text)
    if not match:
        raise RuntimeError(f"previousExchangeSimulationTimeS missing in {path}")
    return float(match.group(1))


def heat_flux_directories(case: Path, interval_s: float = 0.1) -> list[tuple[float, Path]]:
    records: dict[int, tuple[float, Path]] = {}
    for directory_time, directory in numeric_directories(case):
        if directory_time <= 1.0 + TIME_TOLERANCE_S:
            continue
        required = (
            directory / "fluid/gasConvectiveWallHeatFlux",
            directory / "fluid/T",
            directory / "thermalExchangeState",
        )
        if all(foam_file_complete(path) for path in required):
            exchange_time = completed_exchange_time(directory / "thermalExchangeState")
            grid_index = round(exchange_time / interval_s)
            if exchange_time <= 1.0 + TIME_TOLERANCE_S:
                continue
            if abs(exchange_time - grid_index * interval_s) > interval_s * 1.0e-4:
                continue
            records.setdefault(grid_index, (exchange_time, directory))
    return [records[index] for index in sorted(records)]


def time_name(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def configured_end_time(case: Path) -> float:
    text = clean_text(case / "system/controlDict")
    match = re.search(r"\bendTime\s+([-+0-9.eE]+)\s*;", text)
    if not match:
        raise RuntimeError(f"endTime missing in {case / 'system/controlDict'}")
    return float(match.group(1))


def completed_end_time_directory(case: Path) -> Path | None:
    end_time = configured_end_time(case)
    tolerance = max(TIME_TOLERANCE_S, abs(end_time) * 1.0e-8)
    candidates = [
        (abs(time_s - end_time), directory)
        for time_s, directory in numeric_directories(case)
    ]
    if not candidates:
        return None
    error, directory = min(candidates, key=lambda item: item[0])
    return directory if error <= tolerance else None


def bartz_replay_binary() -> Path:
    candidates: list[Path] = []
    for parent in (SCRIPT_CASE, *SCRIPT_CASE.parents):
        candidates.append(parent / "bin/bartzSolidReplay")
        candidates.append(parent / "ugkp_server_update_02/bin/bartzSolidReplay")
    executable = shutil.which("bartzSolidReplay")
    if executable:
        candidates.append(Path(executable))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise RuntimeError("bartzSolidReplay is unavailable")


def replay_bartz_solid_temperature(
    case: Path,
    records: list[tuple[Path, float, np.ndarray]],
) -> int:
    final_directory = completed_end_time_directory(case)
    if final_directory is None:
        print(f"Bartz solid replay pending: endTime has not been written for {case.name}")
        return 0
    if not records or records[-1][0] != final_directory:
        raise RuntimeError("the final written state has no completed Bartz heat-flux interval")
    first_target = float(records[0][0].name)
    snapshots: list[tuple[Path, float, np.ndarray | None]] = [
        (directory, 0.0, None)
        for time_s, directory in numeric_directories(case)
        if time_s < first_target - TIME_TOLERANCE_S
    ]
    snapshots.extend(records)
    schedule_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="bartz-solid-", suffix=".dat", delete=False
        ) as stream:
            schedule_path = Path(stream.name)
            stream.write(f"{len(snapshots)}\n")
            for directory, delta_t, heat_flux in snapshots:
                if heat_flux is None:
                    stream.write(f"{directory.name} {float(directory.name):.17g} 0 0\n")
                    continue
                stream.write(
                    f"{directory.name} {float(directory.name):.17g} "
                    f"{delta_t:.17g} {heat_flux.size}\n"
                )
                stream.writelines(f"{float(value):.17g}\n" for value in heat_flux)
        completed = subprocess.run(
            [
                str(bartz_replay_binary()),
                "-case",
                str(case),
                "-fluxFile",
                str(schedule_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError("Bartz solid replay failed:\n" + completed.stdout[-6000:])
    finally:
        if schedule_path is not None:
            schedule_path.unlink(missing_ok=True)
    output = final_directory / "graphite/Tbartz"
    if not foam_file_complete(output):
        raise RuntimeError(f"Bartz solid replay did not write {output}")
    print(f"Tbartz written: records={len(snapshots)} final={output}")
    return len(snapshots)


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_heat_samples(patch: str = "fluid_to_graphite") -> tuple[np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    if not (BENCHMARK_CASE / "constant/fluid/polyMesh/boundary").is_file():
        return np.empty(0), []
    _, centres = coupled_face_geometry(BENCHMARK_CASE, patch)
    order = np.argsort(centres[:, 0])
    records = []
    for time_s, directory in numeric_directories(BENCHMARK_CASE):
        paths = (directory / "fluid/wallHeatFlux", directory / "fluid/T")
        if not all(foam_file_complete(path) for path in paths):
            continue
        try:
            q = -patch_values(paths[0], patch, len(order))
            tw = patch_values(paths[1], patch, len(order))
        except (RuntimeError, ValueError, OSError):
            continue
        if np.all(np.isfinite(q)) and np.all(np.isfinite(tw)):
            records.append((time_s, q[order], tw[order]))
    return centres[order, 0] * 1000, records


def benchmark_interval(samples: tuple[np.ndarray, list], start: float, end: float, x_mm: np.ndarray) -> np.ndarray:
    x, records = samples
    missing = np.full(x_mm.shape, np.nan)
    if len(records) < 2 or end <= start:
        return missing
    times = np.asarray([item[0] for item in records])
    tolerance = 2.0e-6
    if times[0] > start + tolerance or times[-1] < end - tolerance:
        return missing
    start = max(start, float(times[0]))
    end = min(end, float(times[-1]))
    points = np.unique(np.r_[start, times[(times > start) & (times < end)], end])
    if np.max(np.diff(points)) > 0.011:
        return missing
    q = np.stack([item[1] for item in records])
    interpolated = np.stack([np.interp(points, times, q[:, i]) for i in range(len(x))], axis=1)
    average = np.trapezoid(interpolated, points, axis=0) / (end - start)
    return np.interp(x_mm, x, average, left=np.nan, right=np.nan)


def gas_wall_heat_flux_profiles(case: Path, patch: str = "fluid_to_graphite") -> int:
    if case == BENCHMARK_CASE.resolve():
        return benchmark_wall_heat_flux_profiles(patch)
    _, centres = coupled_face_geometry(case, patch, axisymmetric=True)
    coordinate_mm = centres[:, 0] * 1000.0
    radius = np.sqrt(centres[:, 1] ** 2 + centres[:, 2] ** 2)
    order = np.argsort(coordinate_mm)
    config = json.loads((SHARED_ASSETS / "bartz.json").read_text())
    throat_x = float(config["throat_axial_coordinate_m"])
    pressure_time, pressure = pressure_table(case / "constant/fluid/inletPressure.table")
    output = THERMAL / "results" / case.name
    data_directory = output / "data/wall_heat_flux_profiles"
    figure_directory = output / "figures/wall_heat_flux_profiles"
    for directory in (data_directory, figure_directory):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    records = heat_flux_directories(case)
    benchmark_samples = benchmark_heat_samples(patch)
    if not records:
        raise RuntimeError(f"no completed 0.1 s wall-heat-flux outputs in {case}")
    all_time_directories = [
        (completed_exchange_time(directory / "thermalExchangeState"), directory)
        for _, directory in numeric_directories(case)
        if (directory / "thermalExchangeState").is_file()
    ]
    replay_records: list[tuple[Path, float, np.ndarray]] = []
    for time_s, directory in records:
        actual = patch_values(directory / "fluid/gasConvectiveWallHeatFlux", patch, centres.shape[0])
        wall_temperature_end = patch_values(directory / "fluid/T", patch, centres.shape[0])
        previous_time = previous_exchange_time(directory / "thermalExchangeState")
        benchmark = benchmark_interval(benchmark_samples, previous_time, time_s, coordinate_mm)
        previous_candidates = [
            (abs(candidate_time - previous_time), candidate_directory)
            for candidate_time, candidate_directory in all_time_directories
        ]
        if not previous_candidates:
            raise RuntimeError(f"no preceding wall-temperature state for {directory}")
        previous_error, previous_directory = min(previous_candidates, key=lambda item: item[0])
        if previous_error > TIME_TOLERANCE_S:
            raise RuntimeError(
                f"wall-temperature state {previous_time:.17g} is missing for {directory}"
            )
        wall_temperature, wall_temperature_source = interval_wall_temperature(
            case, previous_directory, patch, centres.shape[0]
        )
        interval_midpoint = 0.5*(previous_time + time_s)
        chamber_pressure = float(np.interp(interval_midpoint, pressure_time, pressure))
        bartz_result = [
            bartz_wall_heat_flux(
                chamber_pressure,
                float(wall_temperature[index]),
                float(radius[index]),
                float(centres[index, 0]),
                throat_x,
                config["total_temperature_k"],
                config["throat_diameter_m"],
                config["throat_curvature_radius_m"],
                config["dynamic_viscosity_pa_s"],
                config["specific_heat_j_kg_k"],
                config["prandtl"],
                config["gamma"],
                config["gas_constant_j_kg_k"],
            )
            for index in range(centres.shape[0])
        ]
        bartz = np.asarray([item[0] for item in bartz_result])
        bartz_h = np.asarray([item[1] for item in bartz_result])
        bartz_mach = np.asarray([item[2] for item in bartz_result])
        replay_records.append((directory, time_s - previous_time, bartz.copy()))
        rows = [
            {
                "time_s": time_s,
                "wall_coordinate_mm": coordinate_mm[index],
                "calculated_wall_heat_flux_W_m2": actual[index],
                "benchmark_W_m2": benchmark[index],
                "benchmark_sampling": "trapezoidal_average_over_same_physical_interval",
                "bartz_W_m2": bartz[index],
                "bartz_h_W_m2_K": bartz_h[index],
                "bartz_mach": bartz_mach[index],
                "local_radius_m": radius[index],
                "wall_temperature_K": wall_temperature[index],
                "wall_temperature_source": wall_temperature_source,
                "wall_temperature_end_K": wall_temperature_end[index],
                "interval_start_time_s": previous_time,
                "interval_midpoint_time_s": interval_midpoint,
            }
            for index in order
        ]
        name = time_name(time_s)
        csv_path = data_directory / f"{name}.csv"
        png_path = figure_directory / f"{name}.png"
        write_rows(csv_path, rows)
        fig, axis = plt.subplots(figsize=(9.4, 6.0), facecolor="white")
        axis.plot(
            coordinate_mm[order],
            actual[order] / 1.0e6,
            color="#1f77b4",
            linewidth=2.2,
            label="Calculated wall heat flux",
        )
        axis.plot(
            coordinate_mm[order],
            bartz[order] / 1.0e6,
            color="#ff7f0e",
            linewidth=2.2,
            linestyle="--",
            label="Bartz",
        )
        if np.any(np.isfinite(benchmark)):
            axis.plot(coordinate_mm[order], benchmark[order] / 1.0e6,
                      color="#2ca02c", linewidth=2, linestyle="-.",
                      label="OpenFOAM SST CHT Benchmark")
        axis.set_xlabel("Axial wall coordinate (mm)")
        axis.set_ylabel(r"Wall heat flux (MW m$^{-2}$)")
        axis.set_title(rf"{case.name.replace('_', ' ')}, $t={time_s:.6f}$ s")
        axis.legend(loc="best", frameon=False)
        axis.grid(alpha=0.18)
        axis.tick_params(top=True, right=True, length=6, width=1.0)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, facecolor="white")
        plt.close(fig)
        shutil.copy2(csv_path, output / "data/wall_heat_flux_profile.csv")
        shutil.copy2(png_path, output / "figures/wall_heat_flux_profile.png")
    replay_bartz_solid_temperature(case, replay_records)
    return len(records)


def benchmark_wall_heat_flux_profiles(patch: str = "fluid_to_graphite") -> int:
    x, samples = benchmark_heat_samples(patch)
    if not samples:
        raise RuntimeError("the OpenFOAM Benchmark has no complete wall-heat-flux output yet")
    output = THERMAL / "results" / BENCHMARK_CASE.name
    data = output / "data/wall_heat_flux_profiles"
    figures = output / "figures/wall_heat_flux_profiles"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _, centres = coupled_face_geometry(BENCHMARK_CASE, patch, axisymmetric=True)
    order = np.argsort(centres[:, 0])
    radius = np.linalg.norm(centres[order, 1:], axis=1)
    config = json.loads((SHARED_ASSETS / "bartz.json").read_text())
    throat_x = float(config["throat_axial_coordinate_m"])
    pt, pv = pressure_table(SST_CASE / "constant/fluid/inletPressure.table")
    count = 0
    for time_s, q, tw in samples:
        if abs(time_s * 10 - round(time_s * 10)) > 1e-6:
            continue
        bartz = np.asarray([
            bartz_wall_heat_flux(float(np.interp(time_s, pt, pv)), float(tw[i]), float(radius[i]),
                                float(x[i] / 1000), throat_x, config["total_temperature_k"],
                                config["throat_diameter_m"], config["throat_curvature_radius_m"],
                                config["dynamic_viscosity_pa_s"], config["specific_heat_j_kg_k"],
                                config["prandtl"], config["gamma"], config["gas_constant_j_kg_k"])[0]
            for i in range(len(x))])
        rows = [{"time_s": time_s, "wall_coordinate_mm": x[i], "benchmark_W_m2": q[i],
                 "bartz_W_m2": bartz[i], "wall_temperature_K": tw[i],
                 "sampling": "instantaneous_at_written_physical_time"} for i in range(len(x))]
        name = time_name(time_s)
        write_rows(data / f"{name}.csv", rows)
        fig, axis = plt.subplots(figsize=(9.4, 6), facecolor="white")
        axis.plot(x, q / 1e6, color="#2ca02c", label="OpenFOAM SST CHT Benchmark")
        axis.plot(x, bartz / 1e6, color="#ff7f0e", linestyle="--", label="Bartz")
        axis.set_xlabel("Axial wall coordinate (mm)")
        axis.set_ylabel(r"Wall heat flux (MW m$^{-2}$)")
        axis.set_title(f"OpenFOAM Benchmark, t={time_s:g} s (instantaneous)")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=300)
        plt.close(fig)
        shutil.copy2(data / f"{name}.csv", output / "data/wall_heat_flux_profile.csv")
        shutil.copy2(figures / f"{name}.png", output / "figures/wall_heat_flux_profile.png")
        count += 1
    return count


def refresh_probes(case: Path) -> None:
    if not any(time_s > 1.0 + TIME_TOLERANCE_S for time_s, _ in numeric_directories(case)):
        return
    command = [
        "bash",
        "-lc",
        "source /opt/openfoam10/etc/bashrc >/dev/null 2>&1 && "
        "exec postProcess -region graphite -dict system/mss7TemperatureProbesDict -time 1:",
    ]
    completed = subprocess.run(command, cwd=case, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("temperature probe extraction failed:\n" + completed.stdout[-4000:])


def refresh_bartz_probes(case: Path) -> None:
    if not any((directory / "graphite/Tbartz").is_file() for _, directory in numeric_directories(case)):
        return
    source = clean_text(case / "system/mss7TemperatureProbesDict")
    function = re.search(r"\bmss7ThroatTemperatureProbes\s*\{(.*?)\n\s*\}", source, re.S)
    if not function:
        raise RuntimeError("mss7ThroatTemperatureProbes is missing")
    locations = re.search(r"\bprobeLocations\s*\((.*?)\)\s*;", function.group(1), re.S)
    if not locations:
        raise RuntimeError("internal probe locations are missing")
    points = re.findall(
        r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",
        locations.group(1),
    )
    if len(points) != 2:
        raise RuntimeError("expected two internal Bartz probe locations")
    dictionary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="mss7-tbartz-probes-", suffix=".dict", delete=False
        ) as stream:
            dictionary_path = Path(stream.name)
            point_text = "\n".join(f"            ({x} {y} {z})" for x, y, z in points)
            stream.write(
                "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
                "    class dictionary;\n    object mss7TBartzProbesDict;\n}\n"
                "functions\n{\n    mss7TBartzTemperatureProbes\n    {\n"
                "        type probes;\n        libs (\"libsampling.so\");\n"
                "        region graphite;\n        fields (Tbartz);\n"
                "        fixedLocations true;\n        interpolationScheme cellPoint;\n"
                "        probeLocations\n        (\n"
                f"{point_text}\n"
                "        );\n        writeControl timeStep;\n        writeInterval 1;\n"
                "    }\n}\n"
            )
        command = (
            "source /opt/openfoam10/etc/bashrc >/dev/null 2>&1 && "
            f"exec postProcess -case {shlex.quote(str(case))} -region graphite "
            f"-dict {shlex.quote(str(dictionary_path))} -time 1:"
        )
        completed = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError("Tbartz probe extraction failed:\n" + completed.stdout[-4000:])
    finally:
        if dictionary_path is not None:
            dictionary_path.unlink(missing_ok=True)


def read_probe_series(
    path: Path,
    field_name: str = "T",
) -> tuple[list[tuple[float, float, float]], dict[float, list[float]]]:
    coordinates: list[tuple[float, float, float]] | None = None
    samples: dict[float, list[float]] = {}
    for _, directory in numeric_directories(path):
        probe_file = directory / field_name
        if not probe_file.is_file():
            continue
        local_coordinates: list[tuple[float, float, float]] = []
        for line in probe_file.read_text(encoding="utf-8").splitlines():
            match = re.match(
                r"#\s*Probe\s+\d+\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)",
                line,
            )
            if match:
                local_coordinates.append(tuple(float(value) for value in match.groups()))
            elif line.strip() and not line.lstrip().startswith("#"):
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
    wall_coordinates, wall_samples = read_probe_series(post / "mss7ThroatWallProbe/graphite")
    internal_coordinates, internal_samples = read_probe_series(post / "mss7ThroatTemperatureProbes/graphite")
    if len(internal_coordinates) != 2:
        raise RuntimeError(f"expected two internal MSS7 probes in {case}")
    result: dict[float, tuple[float, float, float]] = {}
    for time_s, internal_values in internal_samples.items():
        candidates = [candidate for candidate in wall_samples if abs(candidate - time_s) <= TIME_TOLERANCE_S]
        if candidates:
            result[time_s] = (
                wall_temperature(wall_coordinates, wall_samples[candidates[0]], target_wall_x),
                internal_values[0],
                internal_values[1],
            )
    if not result:
        raise RuntimeError(f"no matched wall/internal MSS7 samples in {case}")
    return result


def read_bartz_case(case: Path) -> dict[float, tuple[float, float]]:
    coordinates, samples = read_probe_series(
        case / "postProcessing/mss7TBartzTemperatureProbes/graphite",
        "Tbartz",
    )
    if len(coordinates) != 2:
        raise RuntimeError(f"expected two internal Tbartz probes in {case}")
    result = {
        time_s: (values[0], values[1])
        for time_s, values in samples.items()
        if len(values) == 2 and np.all(np.isfinite(values))
    }
    if not result:
        raise RuntimeError(f"no complete Tbartz probe samples in {case}")
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


def benchmark_temperatures(target_x: float) -> dict[float, tuple[float, float, float]]:
    _, internal = read_probe_series(BENCHMARK_CASE / "postProcessing/mss7ThroatTemperatureProbes/graphite")
    x, heat = benchmark_heat_samples()
    result = {}
    for time_s, _, wall in heat:
        matches = [candidate for candidate in internal if abs(candidate - time_s) <= 2e-6]
        if not matches:
            continue
        values = internal[matches[0]]
        if len(values) != 2 or not np.all(np.isfinite(values)) or any(v <= 0 or v > 1e5 for v in values):
            continue
        if x[0] <= target_x * 1000 <= x[-1]:
            result[time_s] = (float(np.interp(target_x * 1000, x, wall)), values[0], values[1])
    return result


def temperature_comparison() -> int:
    for case in (LAMINAR_CASE, SST_CASE):
        try:
            refresh_probes(case)
        except RuntimeError as error:
            print(f"probe refresh skipped for {case.name}: {error}", file=sys.stderr)
    try:
        refresh_bartz_probes(SST_CASE)
    except RuntimeError as error:
        print(f"Tbartz probe refresh skipped: {error}", file=sys.stderr)
    locations = read_probe_locations()
    target_wall_x = float(locations["Tw"]["x_m"])
    cases = {}
    for name, path in (("laminar", LAMINAR_CASE), ("sst", SST_CASE)):
        try:
            cases[name] = read_case(path, target_wall_x)
        except (OSError, RuntimeError, ValueError) as error:
            cases[name] = {}
            print(f"{name} temperature pending: {error}", file=sys.stderr)
    laminar, sst = cases["laminar"], cases["sst"]
    bartz = {}
    try:
        bartz = read_bartz_case(SST_CASE)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Tbartz temperature pending: {error}", file=sys.stderr)
    benchmark = {}
    if BENCHMARK_CASE.is_dir():
        try:
            benchmark = benchmark_temperatures(target_wall_x)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Benchmark temperature pending: {error}", file=sys.stderr)
    times = sorted(set(round(t, 8) for data in (laminar, sst, benchmark, bartz) for t in data))
    if not times:
        raise RuntimeError("no complete temperature samples are available")
    experiment_time, experiment = read_experiment()
    rows: list[dict[str, float]] = []
    keys = ("Tw", "Tn1", "Tn2")
    for time_s in times:
        row: dict[str, float] = {"time_s": time_s}
        for index, key in enumerate(keys):
            row[f"{key}_experiment_K"] = float(np.interp(time_s, experiment_time, experiment[key]))
            for name, samples in (("laminar", laminar), ("sst", sst), ("benchmark", benchmark)):
                candidates = [candidate for candidate in samples if abs(candidate - time_s) <= 2e-6]
                row[f"{key}_{name}_K"] = samples[candidates[0]][index] if candidates else float("nan")
            bartz_candidates = [candidate for candidate in bartz if abs(candidate - time_s) <= 2e-6]
            row[f"{key}_bartz_K"] = (
                bartz[bartz_candidates[0]][index - 1]
                if key != "Tw" and bartz_candidates
                else float("nan")
            )
        rows.append(row)
    csv_output = COMPARISON_RESULTS / "data/mss7_temperature_laminar_sst_experiment.csv"
    png_output = COMPARISON_RESULTS / "figures/mss7_temperature_laminar_sst_experiment.png"
    write_rows(csv_output, rows)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time_s"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), facecolor="white")
    plot_fields = (("Tn1", 1, r"$T_{N1}$"), ("Tn2", 2, r"$T_{N2}$"))
    for panel, (axis, (key, sample_index, label)) in enumerate(zip(axes, plot_fields)):
        measured = np.asarray([row[f"{key}_experiment_K"] for row in rows])
        laminar_values = np.asarray([row[f"{key}_laminar_K"] for row in rows])
        sst_values = np.asarray([row[f"{key}_sst_K"] for row in rows])
        bartz_values = np.asarray([row[f"{key}_bartz_K"] for row in rows])
        axis.plot(time, measured, color="black", linewidth=2.6, label="Experiment")
        mask = np.isfinite(sst_values)
        axis.plot(time[mask], sst_values[mask], color="#C44E52", linewidth=1.6, marker="o", markersize=4.5,
                  markerfacecolor="white", markeredgewidth=0.9, label="SST wall model")
        mask = np.isfinite(laminar_values)
        axis.plot(time[mask], laminar_values[mask], color="#4C72B0", linewidth=1.6, linestyle="--", marker="s",
                  markersize=3.8, markerfacecolor="white", markeredgewidth=0.9, label="Laminar")
        mask = np.isfinite(bartz_values)
        if np.any(mask):
            axis.plot(time[mask], bartz_values[mask], color="#8172B2", linewidth=2.0,
                      linestyle=":", label="TBARTZ")
        benchmark_time = np.asarray(sorted(benchmark))
        benchmark_values = np.asarray([benchmark[t][sample_index] for t in benchmark_time])
        if benchmark_time.size:
            axis.plot(benchmark_time, benchmark_values, color="#2ca02c", linewidth=1.8,
                      linestyle="-.", label="OpenFOAM SST CHT Benchmark")
        all_values = np.concatenate((measured, laminar_values, sst_values, bartz_values, benchmark_values))
        all_values = all_values[np.isfinite(all_values)]
        span = max(20.0, float(all_values.max() - all_values.min()))
        time_span = max(1.0e-12, float(time[-1] - time[0]))
        axis.set_xlim(float(time[0] - 0.025 * time_span), float(time[-1] + 0.025 * time_span))
        axis.set_ylim(float(all_values.min() - 0.08 * span), float(all_values.max() + 0.08 * span))
        axis.set_xlabel(r"Time, $t$ (s)")
        axis.set_ylabel(r"Temperature, $T$ (K)")
        axis.text(0.72 if panel == 0 else 0.05, 0.08 if panel == 0 else 0.92,
                  f"({chr(97 + panel)}) {label}", transform=axis.transAxes)
        axis.tick_params(top=True, right=True, length=6, width=1.0)
        axis.grid(False)
        if panel == 0:
            axis.legend(loc="best", frameon=False)
    fig.tight_layout(pad=0.7, w_pad=1.0)
    fig.savefig(png_output, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"wrote={csv_output}")
    print(f"wrote={png_output}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=SCRIPT_CASE)
    parser.add_argument("--only", choices=("all", "temperature", "wall-heat-flux"), default="all")
    args = parser.parse_args()
    case = args.case.resolve()
    tasks = []
    if args.only in ("all", "wall-heat-flux"):
        tasks.append((f"{case.name} wall heat flux", lambda: gas_wall_heat_flux_profiles(case)))
    if args.only in ("all", "temperature"):
        tasks.append(("temperature comparison", temperature_comparison))
    for label, task in tasks:
        if args.only == "all":
            try:
                result = task()
                if isinstance(result, int):
                    print(f"{label}: records={result}")
            except (OSError, RuntimeError, ValueError) as error:
                print(f"{label} skipped: {error}", file=sys.stderr)
        else:
            result = task()
            if isinstance(result, int):
                print(f"{label}: records={result}")


if __name__ == "__main__":
    main()
