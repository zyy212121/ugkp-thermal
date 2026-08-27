#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path


FACTORS = (1, 2, 4, 8, 16, 32, 64)
REPEATS = 5
PROBE_INTERVAL = 10
EXPECTED_STEPS = 200
UNIFIED_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("UGKP_HEAVY_LAVAL_INPUTS", UNIFIED_ROOT / "examples" / "performance" / "heavy_laval_inputs"))
ARCHIVE_ROOT = Path(os.environ.get("UGKP_HEAVY_LAVAL_ARCHIVE", UNIFIED_ROOT / "examples" / "performance" / "results" / "heavy_laval"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_factors(value: str) -> tuple[int, ...]:
    try:
        requested = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("factors must be comma-separated integers") from error
    if not requested or len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("factors must be nonempty and unique")
    invalid = tuple(factor for factor in requested if factor not in FACTORS)
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported factors: {invalid}")
    return requested


def scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s+([^;]+);", text)
    if match is None:
        raise ValueError(f"missing scalar {key}")
    return match.group(1).strip()


def replace_scalar(text: str, key: str, value: str) -> str:
    result, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}\s+)[^;]+;",
        rf"\g<1>{value};",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"missing scalar {key}")
    return result


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def convert_weighted_restart(path: Path) -> tuple[str, str]:
    source_hash = sha256(path)
    temporary = path.with_name(path.name + ".tmp")
    with path.open("rb") as source, temporary.open("wb") as destination:
        header = source.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[0] != "GPU3_PARTICLES_V1_BIN":
            raise ValueError(f"{path}: unsupported archived restart header")
        destination.write(f"UGKP_PARTICLES_SCHEMA1_BIN {header[1]} {header[2]}\n".encode("ascii"))
        shutil.copyfileobj(source, destination, length=8 << 20)
    os.replace(temporary, path)
    return source_hash, sha256(path)


def convert_source_residual(path: Path) -> tuple[str, str]:
    source_hash = sha256(path)
    text = path.read_text(encoding="utf-8")
    converted, count = re.subn(r"\AGPU2_SOURCE_RESIDUAL_V1\b", "UGKP_SOURCE_RESIDUAL_SCHEMA1", text, count=1)
    if count != 1:
        raise ValueError(f"{path}: unsupported archived source residual header")
    write_atomic(path, converted)
    return source_hash, sha256(path)


def make_fluid_properties(source: Path) -> str:
    physical = source.read_text(encoding="utf-8")
    physical = re.sub(r"(?m)^\s*object\s+physicalProperties\s*;", "    object      fluidProperties;", physical, count=1)
    marker = physical.find("thermoType")
    if marker < 0:
        raise ValueError(f"{source}: thermoType missing")
    head = physical[:marker]
    body = physical[marker:]
    if "schemaVersion" not in head:
        head += "\nschemaVersion 1;\n\n"
    return head + body.rstrip() + "\n\nturbulence\n{\n    simulationType  laminar;\n}\n"


def make_particle_properties(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*object\s+ugkwpProperties\s*;", "    object      particleProperties;", text, count=1)
    closing = text.find("// * * *")
    if closing < 0:
        raise ValueError(f"{source}: header delimiter missing")
    text = text[:closing] + "schemaVersion 1;\n\n" + text[closing:]
    for key in (
        "gpuCsrHeavyCellThreshold",
        "gpuCsrHeavyTileParticles",
        "gpuCsrHeavyWorkerBlocksPerSM",
        "gpuCsrCellLocalPath",
        "gpuCsrWarpAggregatedBinning",
        "gpuCsrSplitPreDirectory",
        "gpuCsrHeavyReduction",
        "gpuCsrHeavyReductionAutoInterval",
        "gpuCsrLevel",
        "gpuResidentStrict",
        "bn",
    ):
        text = re.sub(rf"(?m)^\s*{key}\s+[^;]+;\s*\n?", "", text, count=1)
    return text


def make_scheduling_properties(particle_capacity: int) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       dictionary;\n"
        "    location    \"constant\";\n"
        "    object      schedulingProperties;\n"
        "}\n\n"
        "schemaVersion 1;\n\n"
        "gpuResidentPureGasOnly false;\n"
        "gpuResidentDynamicInlet true;\n"
        f"gpuResidentParticleCapacity {particle_capacity};\n"
        "gpuResidentMaxFaceWalkHops 12;\n"
        "gpuResidentCourantUpdateInterval 100;\n"
        "gpuResidentMaxDeltaTGrowth 1.05;\n"
        "gpuCsrLevel L2;\n"
        "gpuParticleBlockThreads 128;\n"
        "gpuReductionBlockThreads 128;\n"
    )


def prepare(output_root: Path, factors: tuple[int, ...]) -> None:
    inputs = output_root / "inputs"
    if inputs.exists():
        raise FileExistsError(inputs)
    rows = []
    for factor in factors:
        source = SOURCE_ROOT / f"k{factor}"
        destination = inputs / f"k{factor}"
        copy_tree(source, destination)
        old_manifest = json.loads((source / "case_manifest.json").read_text(encoding="utf-8"))
        restart_relative = Path(str(old_manifest["start_time_directory"])) / "gpuResidentStrictParticles.dat"
        restart = destination / restart_relative
        archived_restart_hash = sha256(restart)
        if archived_restart_hash != old_manifest["restart_sha256"]:
            raise ValueError(f"k{factor}: restart hash changed during copy")
        source_restart_hash, restart_hash = convert_weighted_restart(restart)
        if source_restart_hash != archived_restart_hash:
            raise ValueError(f"k{factor}: archived restart identity changed")
        source_residual = destination / str(old_manifest["start_time_directory"]) / "gpuResidentStrictSourceResidual.dat"
        archived_residual_hash, residual_hash = convert_source_residual(source_residual)
        fluid = make_fluid_properties(destination / "constant" / "physicalProperties")
        particle = make_particle_properties(destination / "constant" / "ugkwpProperties")
        write_atomic(destination / "constant" / "fluidProperties", fluid)
        write_atomic(destination / "constant" / "particleProperties", particle)
        write_atomic(
            destination / "constant" / "schedulingProperties",
            make_scheduling_properties(int(old_manifest["particle_capacity"])),
        )
        (destination / "constant" / "physicalProperties").unlink()
        (destination / "constant" / "momentumTransport").unlink()
        (destination / "constant" / "ugkwpProperties").unlink()
        control = (destination / "system" / "controlDict").read_text(encoding="utf-8")
        control = replace_scalar(control, "application", "gasUGKP")
        write_atomic(destination / "system" / "controlDict", control)
        manifest = {
            "schema": "unified-heavy-laval-l2-input-v1",
            "factor": factor,
            "target_cell": int(old_manifest["target_cell"]),
            "selected_source_particles": int(old_manifest["selected_source_particles"]),
            "particles": int(old_manifest["particles"]),
            "particle_capacity": int(old_manifest["particle_capacity"]),
            "start_time_directory": str(old_manifest["start_time_directory"]),
            "steps": int(old_manifest["steps"]),
            "delta_t": float(old_manifest["delta_t"]),
            "restart_sha256": restart_hash,
            "archived_restart_sha256": archived_restart_hash,
            "source_residual_sha256": residual_hash,
            "archived_source_residual_sha256": archived_residual_hash,
            "legacy_manifest_sha256": sha256(source / "case_manifest.json"),
            "fluid_properties_sha256": sha256(destination / "constant" / "fluidProperties"),
            "particle_properties_sha256": sha256(destination / "constant" / "particleProperties"),
            "scheduling_properties_sha256": sha256(destination / "constant" / "schedulingProperties"),
            "fv_schemes_sha256": sha256(destination / "system" / "fvSchemes"),
            "control_dict_sha256": sha256(destination / "system" / "controlDict"),
        }
        if manifest["steps"] != EXPECTED_STEPS or abs(manifest["delta_t"] - 5e-8) > 1e-20:
            raise ValueError(f"k{factor}: timing workload changed")
        (destination / "unified_l2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "input_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_case(case: Path, manifest: dict[str, object]) -> None:
    particle_path = case / "constant" / "particleProperties"
    scheduling_path = case / "constant" / "schedulingProperties"
    fluid_path = case / "constant" / "fluidProperties"
    control_path = case / "system" / "controlDict"
    if sha256(particle_path) != manifest["particle_properties_sha256"] or sha256(scheduling_path) != manifest["scheduling_properties_sha256"] or sha256(fluid_path) != manifest["fluid_properties_sha256"] or sha256(control_path) != manifest["control_dict_sha256"]:
        raise ValueError("prepared dictionaries changed")
    properties = particle_path.read_text(encoding="utf-8")
    scheduling = scheduling_path.read_text(encoding="utf-8")
    required = {
        "gpuCsrLevel": "L2",
        "gpuParticleBlockThreads": "128",
        "gpuReductionBlockThreads": "128",
    }
    for key, value in required.items():
        if scalar(scheduling, key) != value:
            raise ValueError(f"unexpected {key}")
    parcel_mass = Decimal(scalar(properties, "parcelMass"))
    injection_mass = Decimal(scalar(properties, "injectionParcelMass"))
    restart_mass = Decimal(scalar(properties, "legacyRestartParcelMass"))
    expected_mass = Decimal("1.0000000000000001e-11")
    if parcel_mass != expected_mass or injection_mass != expected_mass or restart_mass != expected_mass:
        raise ValueError("forced parcel mass changed")
    if scalar((case / "system" / "fvSchemes").read_text(encoding="utf-8"), "fluxScheme") != "Tadmor":
        raise ValueError("legacy flux scheme changed")
    restart = case / str(manifest["start_time_directory"]) / "gpuResidentStrictParticles.dat"
    if sha256(restart) != manifest["restart_sha256"]:
        raise ValueError("restart changed")
    residual = case / str(manifest["start_time_directory"]) / "gpuResidentStrictSourceResidual.dat"
    if sha256(residual) != manifest["source_residual_sha256"]:
        raise ValueError("source residual changed")


def validate_probe(path: Path, run_id: str) -> dict[str, float | int]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_STEPS // PROBE_INTERVAL:
        raise ValueError(f"{run_id}: expected 20 timing samples, found {len(rows)}")
    stages = [name for name in rows[0] if name.endswith("_ms")]
    for row in rows:
        if row["run_id"] != run_id or row["status"] != "ok" or row["timing_valid"] != "1":
            raise ValueError(f"{run_id}: invalid timing identity/status")
        if int(row["heavy_reduction_enabled"]) != 1 or int(row["particle_path"]) != 1:
            raise ValueError(f"{run_id}: L2 was not active")
        if int(row["base_particle_count"]) + int(row["injected_particle_count"]) != int(row["pretransport_particle_count"]):
            raise ValueError(f"{run_id}: pretransport population identity failed")
        if int(row["pretransport_particle_count"]) - int(row["removed_particle_count"]) != int(row["particle_count"]):
            raise ValueError(f"{run_id}: final population identity failed")
        if int(row["particle_count"]) != int(row["occupancy_sum"]) or row["occupancy_matches_count"] != "1":
            raise ValueError(f"{run_id}: occupancy identity failed")
        if any(int(row[key], 0) != 0 for key in ("bad_cells", "bad_particles", "bad_field_mask")):
            raise ValueError(f"{run_id}: invalid particle state")
        values = [float(row[name]) for name in stages]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"{run_id}: invalid timing value")
    result: dict[str, float | int] = {name: statistics.median(float(row[name]) for row in rows) for name in stages}
    for name in ("particle_count", "occupancy_max", "heavy_cell_count", "heavy_particle_count", "heavy_task_count_estimate"):
        result[name] = int(statistics.median(int(row[name]) for row in rows))
    return result


def run_matrix(
    output_root: Path,
    repeats: int,
    factors: tuple[int, ...],
    resume: bool = False,
) -> None:
    frontend = Path(shutil.which("gasUGKP") or "")
    backend = Path(shutil.which("gasUGKPCudaBackend") or "")
    if not frontend.is_file() or not backend.is_file():
        raise FileNotFoundError("installed unified gasUGKP frontend/backend not found")
    runs = output_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    schedule = []
    for factor in factors:
        for repetition in range(repeats + 1):
            schedule.append((factor, repetition, repetition > 0))
    with (output_root / "schedule.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "factor", "repetition", "measured"))
        for sequence, (factor, repetition, measured) in enumerate(schedule, 1):
            writer.writerow((sequence, factor, repetition, int(measured)))
    binary = {
        "frontend": str(frontend),
        "frontend_sha256": sha256(frontend),
        "backend": str(backend),
        "backend_sha256": sha256(backend),
    }
    (output_root / "binary_manifest.json").write_text(json.dumps(binary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_root / "process_medians.csv"
    summaries = rows(summary_path) if resume and summary_path.is_file() else []
    completed_run_ids = {row["run_id"] for row in summaries}
    for sequence, (factor, repetition, measured) in enumerate(schedule, 1):
        case = output_root / "inputs" / f"k{factor}"
        manifest = json.loads((case / "unified_l2_manifest.json").read_text(encoding="utf-8"))
        validate_case(case, manifest)
        run_id = f"k{factor}_{'rep' if measured else 'warmup'}{repetition:02d}_L2"
        if run_id in completed_run_ids:
            print(f"[{sequence}/{len(schedule)}] {run_id} already validated", flush=True)
            continue
        run_dir = runs / run_id
        if run_dir.exists():
            if not resume:
                raise FileExistsError(run_dir)
            shutil.rmtree(run_dir)
        run_dir.mkdir()
        environment = dict(os.environ)
        environment.update({
            "GAS_UGKP_CUDA_BACKEND": str(backend),
            "UGKP_DEV_PROBE_MODE": "timing",
            "UGKP_DEV_PROBE_LOG": str(run_dir / "stage_timing.csv"),
            "UGKP_DEV_PROBE_INTERVAL": str(PROBE_INTERVAL),
            "UGKP_DEV_PROBE_RUN_ID": run_id,
            "UGKP_DEV_PROBE_VARIANT": f"K{factor}_L2_B64",
            "UGKP_DEV_PROBE_FAIL_ON_NONFINITE": "1",
        })
        started = time.time()
        with (run_dir / "solver.log").open("wb") as log:
            completed = subprocess.run([str(frontend), "-case", str(case)], stdout=log, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode != 0:
            raise RuntimeError(f"{run_id}: solver failed with {completed.returncode}")
        summary = validate_probe(run_dir / "stage_timing.csv", run_id)
        validate_case(case, manifest)
        summary.update({
            "sequence": sequence,
            "factor": factor,
            "repetition": repetition,
            "measured": int(measured),
            "run_id": run_id,
            "wall_seconds": time.time() - started,
            "restart_sha256": manifest["restart_sha256"],
            "frontend_sha256": binary["frontend_sha256"],
            "backend_sha256": binary["backend_sha256"],
        })
        (run_dir / "validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
        with summary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"[{sequence}/{len(schedule)}] {run_id} total_ms={summary['total_ms']:.6g} wall_s={summary['wall_seconds']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--factors", type=parse_factors, default=FACTORS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.command == "prepare":
        output.mkdir(parents=True, exist_ok=False)
        prepare(output, args.factors)
    else:
        run_matrix(output, args.repeats, args.factors, args.resume)


if __name__ == "__main__":
    main()
