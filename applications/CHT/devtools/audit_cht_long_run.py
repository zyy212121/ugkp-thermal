#!/usr/bin/env python3
                                                                          

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

DEVTOOLS = Path(__file__).resolve().parent
if str(DEVTOOLS) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS))

from cht_v2_contract import (
    PERFORMANCE_MIE_SHA256,
    case_contracts,
    parse_single_scalar,
    resolve_single_mie_table,
)
from compare_cht_particle_restart import CONTINUOUS, read_restart
from generate_cht_perf_particles import read_cell_volumes
from summarize_cht_performance import (
    EXPECTED_BINARY_SHA256,
    parse_manifest,
    require_recorded_hash,
    tree_sha256,
    validate_runtime_log,
)


TEMPERATURE_BINS = (0.0, 300.0, 500.0, 750.0, 900.0, 975.0, 1000.0, 1025.0, 1100.0, 1250.0, 1500.0, 2000.0, math.inf)
SPEED_BINS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf)
NONFINITE = re.compile(
    r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
LIFECYCLE_TRUE = (
    "gasWallLedgerConsumed",
    "particleWallLedgerConsumed",
    "particleRadiationApplied",
    "solidStateUpdated",
    "wallTemperatureUploaded",
    "particleMomentsRebuilt",
)
EXPECTED_START_TIME = "0.10000002"
EXPECTED_FINAL_TIME = "0.1000013"
EXPECTED_PARCEL_MASS = 5.0e-9
EXPECTED_PARTICLE_CP = 1500.0
EXPECTED_INPUT_TREE_SHA256 = "5055925cd751e0aa3b793691d145f29744f5c7b9c01705eddf5a91f1aa18ca5a"
EXPECTED_START_TREE_SHA256 = "b22e25dc0f29c0f35d7caa4e80d54e4c72c70516bbd6a83430da50836a10f0d7"
EXPECTED_CONTROL_SHA256 = "666b7c2dd36d8a67b89638c1b2a4ddf969758bd43e6a608107030ebc023d8fa1"
EXPECTED_COUPLING_SHA256 = "f529024026d7cdbb8dbc98aaaae3619084db699b8715812a0cc8fad5999da6eb"
EXPECTED_THERMAL_STATE_SHA256 = "6c236fc814877a68d69f8ced53e807f94638f07f26b8586be6e8ad43841a5022"
EXPECTED_VOLUME_SHA256 = {
    "fluid": "435a81c1cb8e45a505bd9560ea2f2bb1613578614791878a5b86ac67721eff42",
    "solid1": "af413773bea116446def77be916aabfb38906c48e32529fd3c0360bac993ec24",
    "solid2": "92e006ab1dfd1c1c4e29bb84936e2dfa0f29639f4ce3edc6dba1f6f586e4d74a",
}
EXPECTED_LONG_RUNS = {
    "baseline": {
        "label": "long-128-baseline-attempt2",
        "case_name": "20260718-cht-long-128-baseline-attempt2",
        "binary_sha256": EXPECTED_BINARY_SHA256["baseline"],
        "manifest_sha256": "0d6af2893eb24868be3a1fc190a65f22b9204cab3c5a85e7efc5354837f47ea8",
        "solver_log_sha256": "c3ac7189b926019558fb5369ed99890affd93f6b257c63cfc52270ab91feb89d",
        "runtime_log_sha256": "e2c5ae7db3144d4e4bb0ca0cf0c3e2d6cf402f8ffa749d9a8dc3ec63ee1725f0",
        "final_tree_sha256": "e057407330e55c1c9c4f03d28b8e2e206efb64385d20d48db88233419c6af20f",
    },
    "baseline_repeat": {
        "label": "long-128-baseline-repeat-attempt2",
        "case_name": "20260718-cht-long-128-baseline-repeat-attempt2",
        "binary_sha256": EXPECTED_BINARY_SHA256["baseline"],
        "manifest_sha256": "f25b62acd3b6af3e1c34697e6b78809397e143707f85a01aef79afd07ddeb0d8",
        "solver_log_sha256": "86e5d361ab7efbe609c40cfd090b2c382eca5d1cc60164d6bf34e89d00a8852f",
        "runtime_log_sha256": "cc783c3fc7ec57075901f5e6367529a054f0bc9a43b48b44e8e8cc7fff7074dc",
        "final_tree_sha256": "a2f3939b6d0d8ea613ba65416b63702f2b7cdf0af67bd75e501a8a68377be8ef",
    },
    "candidate": {
        "label": "long-128-v2-attempt2",
        "case_name": "20260718-cht-long-128-v2-attempt2",
        "binary_sha256": EXPECTED_BINARY_SHA256["v2"],
        "manifest_sha256": "146418de86b275c63b6f3747132779724410259f5ba4f71658bdd32e66b302f4",
        "solver_log_sha256": "e23a334b02acaf8470133aacc42bb1bdd9ffc10ce4c1bc148c32c2e2dddb6fad",
        "runtime_log_sha256": "de1cfb19b6e6733e776f4451e05107781b741b809c0fbec6ff43fd8e0b1531b7",
        "final_tree_sha256": "f4ba31218ebf74f6f839b72a46c78e42e3907a3c6da587654f6d959c38f26d66",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def particle_values_are_finite(values: dict[str, float | int]) -> bool:
    return all(math.isfinite(float(values[name])) for name in CONTINUOUS)


def validate_physical_parameters(parcel_mass: float, particle_cp: float) -> None:
    if not math.isfinite(parcel_mass) or parcel_mass != EXPECTED_PARCEL_MASS:
        raise ValueError(
            f"frozen long run requires parcelMass={EXPECTED_PARCEL_MASS:.17g}, got {parcel_mass!r}"
        )
    if not math.isfinite(particle_cp) or particle_cp != EXPECTED_PARTICLE_CP:
        raise ValueError(
            f"frozen long run requires particleCp={EXPECTED_PARTICLE_CP:.17g}, got {particle_cp!r}"
        )


def validate_case_physical_parameters(case_dir: Path) -> None:
    parcel_values = [
        parse_single_scalar(case_dir / "constant" / relative, "parcelMass")
        for relative in ("ugkwpProperties", "fluid/ugkwpProperties")
    ]
    cp_values = [
        parse_single_scalar(case_dir / "constant" / relative, "particleCp")
        for relative in ("gksProperties", "fluid/gksProperties")
    ]
    if any(value != EXPECTED_PARCEL_MASS for value in parcel_values):
        raise ValueError(f"long-run parcelMass mirrors are not frozen: {parcel_values}")
    if any(value != EXPECTED_PARTICLE_CP for value in cp_values):
        raise ValueError(f"long-run particleCp mirrors are not frozen: {cp_values}")


def validate_distinct_run_inputs(
    run_inputs: dict[str, tuple[Path, Path]],
) -> None:
    if len({case.resolve() for case, _ in run_inputs.values()}) != len(run_inputs):
        raise ValueError("baseline, repeat, and candidate case paths must be distinct")
    if len({log.resolve() for _, log in run_inputs.values()}) != len(run_inputs):
        raise ValueError("baseline, repeat, and candidate solver-log paths must be distinct")


def validate_long_run_identity(
    role: str,
    case_dir: Path,
    solver_log: Path,
    start_time: str,
    final_time: str,
) -> dict[str, str]:
    expected = EXPECTED_LONG_RUNS[role]
    if start_time != EXPECTED_START_TIME or final_time != EXPECTED_FINAL_TIME:
        raise ValueError(
            f"frozen long-run times must be {EXPECTED_START_TIME}->{EXPECTED_FINAL_TIME}"
        )
    if case_dir.name != expected["case_name"]:
        raise ValueError(
            f"{role} case identity mismatch: expected {expected['case_name']}, got {case_dir.name}"
        )
    if solver_log.name != "solver.log" or solver_log.parent.name != expected["label"]:
        raise ValueError(f"{role} solver-log identity mismatch: {solver_log}")

    manifest = solver_log.parent / "manifest.txt"
    runtime_log = solver_log.parent / "runtime.txt"
    frozen_files = {
        "manifest_sha256": (manifest, expected["manifest_sha256"]),
        "solver_log_sha256": (solver_log, expected["solver_log_sha256"]),
        "runtime_log_sha256": (runtime_log, expected["runtime_log_sha256"]),
    }
    identity: dict[str, str] = {}
    for key, (path, expected_hash) in frozen_files.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{role} {path.name} fingerprint mismatch: expected {expected_hash}, got {actual_hash}"
            )
        identity[key] = actual_hash

    fields, hashes = parse_manifest(manifest)
    if fields.get("label") != expected["label"]:
        raise ValueError(f"{role} manifest label mismatch: {fields.get('label')!r}")
    if Path(fields.get("case", "")).resolve() != case_dir.resolve():
        raise ValueError(f"{role} manifest/case path mismatch")
    binary = Path(fields.get("binary", ""))
    require_recorded_hash(hashes, binary, expected["binary_sha256"], f"{role} executable")
    if fields.get("solver_exit_code") != "0":
        raise ValueError(f"{role} manifest does not record solver exit 0")
    validate_runtime_log(runtime_log)
    validate_solver_log(solver_log)
    solver_text = solver_log.read_text(encoding="utf-8")
    runtime_text = runtime_log.read_text(encoding="utf-8")
    resolved_case = case_dir.resolve()
    if (
        f"Exec   : {binary} -case {resolved_case}" not in solver_text
        or f"Case   : {resolved_case}" not in solver_text
        or f'Command being timed: "{binary} -case {resolved_case}"' not in runtime_text
    ):
        raise ValueError(f"{role} logs do not bind the manifest binary/case command")

    contract_failures = case_contracts(
        case_dir, "fallback", require_fresh=False, require_prepared=True
    )
    if contract_failures:
        raise ValueError(f"{role} case contract failed: " + "; ".join(contract_failures))
    validate_case_physical_parameters(case_dir)
    mie_table = resolve_single_mie_table(
        case_dir, case_dir / "constant/solidThermalCouplingProperties"
    )
    mie_table_hash = sha256(mie_table)
    if mie_table_hash != PERFORMANCE_MIE_SHA256:
        raise ValueError(
            f"{role} Mie-table fingerprint mismatch: "
            f"expected {PERFORMANCE_MIE_SHA256}, got {mie_table_hash}"
        )
    if sha256(case_dir / "system/controlDict") != EXPECTED_CONTROL_SHA256:
        raise ValueError(f"{role} controlDict fingerprint mismatch")
    if sha256(case_dir / "constant/solidThermalCouplingProperties") != EXPECTED_COUPLING_SHA256:
        raise ValueError(f"{role} coupling-properties fingerprint mismatch")

    input_tree = tree_sha256(case_dir, ("system", "constant", start_time))
    start_tree = tree_sha256(case_dir / start_time)
    final_tree = tree_sha256(case_dir / final_time)
    if input_tree != EXPECTED_INPUT_TREE_SHA256:
        raise ValueError(f"{role} full input snapshot mismatch: {input_tree}")
    if start_tree != EXPECTED_START_TREE_SHA256:
        raise ValueError(f"{role} initial-state snapshot mismatch: {start_tree}")
    if final_tree != expected["final_tree_sha256"]:
        raise ValueError(f"{role} final-state snapshot mismatch: {final_tree}")
    identity.update(
        {
            "label": expected["label"],
            "case": str(case_dir),
            "binary_sha256": expected["binary_sha256"],
            "input_tree_sha256": input_tree,
            "start_tree_sha256": start_tree,
            "final_tree_sha256": final_tree,
            "control_sha256": EXPECTED_CONTROL_SHA256,
            "coupling_sha256": EXPECTED_COUPLING_SHA256,
            "mie_table_sha256": mie_table_hash,
        }
    )
    return identity


def read_internal_field(path: Path, components: int, expected: int) -> list[float | tuple[float, ...]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        marker = next(i for i, line in enumerate(lines) if "internalField" in line)
    except StopIteration as exc:
        raise ValueError(f"missing internalField in {path}") from exc
    marker_text = lines[marker]
    if "uniform" in marker_text and "nonuniform" not in marker_text:
        value_text = marker_text.split("uniform", 1)[1].split(";", 1)[0].strip()
        if components == 1:
            value: float | tuple[float, ...] = float(value_text)
        else:
            words = value_text.strip("() ").split()
            if len(words) != components:
                raise ValueError(f"invalid uniform vector in {path}: {value_text!r}")
            value = tuple(float(word) for word in words)
        values = [value] * expected
    else:
        count_index = marker + 1
        while count_index < len(lines) and not lines[count_index].strip():
            count_index += 1
        count = int(lines[count_index].strip())
        list_index = count_index + 1
        while list_index < len(lines) and lines[list_index].strip() != "(":
            list_index += 1
        values = []
        for line in lines[list_index + 1 :]:
            stripped = line.strip()
            if stripped == ")":
                break
            if components == 1:
                values.append(float(stripped))
            else:
                words = stripped.strip("() ").split()
                if len(words) != components:
                    raise ValueError(f"invalid vector row in {path}: {line!r}")
                values.append(tuple(float(word) for word in words))
        if count != len(values):
            raise ValueError(f"field count mismatch in {path}: expected {count}, got {len(values)}")
    if len(values) != expected:
        raise ValueError(f"mesh/field count mismatch in {path}: expected {expected}, got {len(values)}")
    flattened = (
        [float(value) for value in values]
        if components == 1
        else [component for value in values for component in value]                            
    )
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError(f"non-finite internal field value in {path}")
    return values


def histogram(value: float, edges: tuple[float, ...]) -> int:
    for index, upper in enumerate(edges[1:]):
        if value < upper:
            return index
    raise AssertionError("infinite histogram edge missing")


def parse_dictionary_scalar(path: Path, key: str) -> float:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        rf"\b{re.escape(key)}\s+\"?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\"?\s*;",
        text,
    )
    if len(matches) != 1:
        raise ValueError(f"expected one {key} in {path}, got {len(matches)}")
    return float(matches[0])


def particle_metrics(
    restart: Path,
    initial_ids: set[int],
    parcel_mass: float,
    particle_cp: float,
    n_cells: int,
) -> dict[str, object]:
    _, particles = read_restart(restart)
    status_counts = {"inactive": 0, "active": 0, "transient": 0, "other": 0}
    occupancy = [0] * n_cells
    temperature_histogram = [0] * (len(TEMPERATURE_BINS) - 1)
    speed_histogram = [0] * (len(SPEED_BINS) - 1)
    momentum_components = [[], [], []]
    kinetic_terms = []
    thermal_terms = []
    outliers = 0
    temperatures = []
    speeds = []

    for values in particles.values():
        if not particle_values_are_finite(values):
            outliers += 1
            continue
        status = int(values["status"])
        status_key = {0: "inactive", 1: "active", 2: "transient"}.get(status, "other")
        status_counts[status_key] += 1
        cell = int(values["cell"])
        if not 0 <= cell < n_cells:
            outliers += 1
        else:
            occupancy[cell] += 1
        velocity = tuple(float(values[name]) for name in ("ux", "uy", "uz"))
        temperature = float(values["temperature"])
        speed = math.sqrt(sum(value * value for value in velocity))
        if not math.isfinite(speed):
            outliers += 1
            continue
        if not 1.0e-12 <= temperature <= 1.0e30:
            outliers += 1
        for component, value in zip(momentum_components, velocity):
            component.append(parcel_mass * value)
        kinetic_terms.append(0.5 * parcel_mass * speed * speed)
        thermal_terms.append(parcel_mass * particle_cp * temperature)
        temperatures.append(temperature)
        speeds.append(speed)
        temperature_histogram[histogram(temperature, TEMPERATURE_BINS)] += 1
        speed_histogram[histogram(speed, SPEED_BINS)] += 1
    ids = set(particles)
    return {
        "count": len(particles),
        "id_set_sha256": hashlib.sha256(
            "\n".join(str(value) for value in sorted(ids)).encode("ascii")
        ).hexdigest(),
        "status_counts": status_counts,
        "exited": len(initial_ids - ids),
        "unexpected_ids": len(ids - initial_ids),
        "cell_occupancy": occupancy,
        "temperature_histogram": temperature_histogram,
        "temperature_bin_edges": list(TEMPERATURE_BINS[:-1]) + ["inf"],
        "speed_histogram": speed_histogram,
        "speed_bin_edges": list(SPEED_BINS[:-1]) + ["inf"],
        "temperature_min": min(temperatures),
        "temperature_max": max(temperatures),
        "speed_min": min(speeds),
        "speed_max": max(speeds),
        "mass": len(particles) * parcel_mass,
        "momentum": [math.fsum(component) for component in momentum_components],
        "thermal_energy": math.fsum(thermal_terms),
        "kinetic_energy": math.fsum(kinetic_terms),
        "outlier_count": outliers,
    }


def thermal_state_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    flags = {
        key: bool(re.search(rf"\b{key}\s+true\s*;", text))
        for key in LIFECYCLE_TRUE
    }
    return {
        "sha256": sha256(path),
        "exchange_sequence": int(parse_dictionary_scalar(path, "exchangeSequence")),
        "completed_time_index": int(parse_dictionary_scalar(path, "completedTimeIndex")),
        "completed_simulation_time": parse_dictionary_scalar(path, "completedSimulationTimeS"),
        "previous_exchange_time": parse_dictionary_scalar(path, "previousExchangeSimulationTimeS"),
        "particle_contact_energy_j": parse_dictionary_scalar(path, "particleContactEnergyJ"),
        "flags": flags,
    }


def validate_solver_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(" CHTdeltaT = ") != 1:
        raise ValueError(f"expected one CHT exchange in {path}")
    if re.search(r"FOAM FATAL|CUDA error|segmentation fault", text, flags=re.IGNORECASE):
        raise ValueError(f"fatal solver text in {path}")
    if NONFINITE.search(text):
        raise ValueError(f"non-finite solver text in {path}")


def case_metrics(
    case_dir: Path,
    final_time: str,
    fluid_volumes: list[float],
    solid1_volumes: list[float],
    solid2_volumes: list[float],
    initial_ids: set[int],
    parcel_mass: float,
    particle_cp: float,
) -> dict[str, object]:
    final = case_dir / final_time
    particles = particle_metrics(
        final / "gpuResidentStrictParticles.dat",
        initial_ids,
        parcel_mass,
        particle_cp,
        len(fluid_volumes),
    )
    rho = read_internal_field(final / "fluid/rho", 1, len(fluid_volumes))
    rho_u = read_internal_field(final / "fluid/rhoU", 3, len(fluid_volumes))
    rho_e = read_internal_field(final / "fluid/rhoE", 1, len(fluid_volumes))
    solid1_t = read_internal_field(final / "solid1/T", 1, len(solid1_volumes))
    solid2_t = read_internal_field(final / "solid2/T", 1, len(solid2_volumes))

    gas_mass = math.fsum(float(value) * volume for value, volume in zip(rho, fluid_volumes))
    gas_momentum = [
        math.fsum(float(value[axis]) * volume for value, volume in zip(rho_u, fluid_volumes))                       
        for axis in range(3)
    ]
    gas_energy = math.fsum(float(value) * volume for value, volume in zip(rho_e, fluid_volumes))
    solid1_energy = math.fsum(
        1940.0 * 1353.0 * float(value) * volume
        for value, volume in zip(solid1_t, solid1_volumes)
    )
    solid2_energy = math.fsum(
        930.0 * 1340.0 * float(value) * volume
        for value, volume in zip(solid2_t, solid2_volumes)
    )
    particle_momentum = [float(value) for value in particles["momentum"]]                          
    signature = {
        "gas_mass": gas_mass,
        "particle_mass": float(particles["mass"]),
        "total_mass": gas_mass + float(particles["mass"]),
        "gas_momentum_x": gas_momentum[0],
        "gas_momentum_y": gas_momentum[1],
        "gas_momentum_z": gas_momentum[2],
        "particle_momentum_x": particle_momentum[0],
        "particle_momentum_y": particle_momentum[1],
        "particle_momentum_z": particle_momentum[2],
        "total_momentum_x": gas_momentum[0] + particle_momentum[0],
        "total_momentum_y": gas_momentum[1] + particle_momentum[1],
        "total_momentum_z": gas_momentum[2] + particle_momentum[2],
        "gas_energy": gas_energy,
        "particle_thermal_energy": float(particles["thermal_energy"]),
        "particle_kinetic_energy": float(particles["kinetic_energy"]),
        "solid1_thermal_energy": solid1_energy,
        "solid2_thermal_energy": solid2_energy,
        "composite_energy": (
            gas_energy
            + float(particles["thermal_energy"])
            + float(particles["kinetic_energy"])
            + solid1_energy
            + solid2_energy
        ),
    }
    time_file = final / "uniform/time"
    return {
        "case": str(case_dir),
        "final_time_value": parse_dictionary_scalar(time_file, "value"),
        "executed_steps": int(parse_dictionary_scalar(time_file, "index")),
        "delta_t": parse_dictionary_scalar(time_file, "deltaT"),
        "signature": signature,
        "particles": particles,
        "thermal_state": thermal_state_metrics(final / "thermalExchangeState"),
    }


def within_envelope(reference: float, repeat: float, candidate: float) -> tuple[bool, float, float]:
    if not all(math.isfinite(value) for value in (reference, repeat, candidate)):
        return False, math.inf, 0.0
    repeat_delta = abs(reference - repeat)
    candidate_delta = abs(reference - candidate)
    scale = max(1.0, abs(reference), abs(candidate))
    allowed = max(2.0e-9 * scale, 5.0 * repeat_delta)
    return candidate_delta <= allowed, candidate_delta, allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-repeat", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--baseline-repeat-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--start-time", default="0.10000002")
    parser.add_argument("--final-time", default="0.1000013")
    parser.add_argument("--parcel-mass", type=float, default=5e-9)
    parser.add_argument("--particle-cp", type=float, default=1500.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validate_physical_parameters(args.parcel_mass, args.particle_cp)
    run_inputs = {
        "baseline": (args.baseline, args.baseline_log),
        "baseline_repeat": (args.baseline_repeat, args.baseline_repeat_log),
        "candidate": (args.candidate, args.candidate_log),
    }
    validate_distinct_run_inputs(run_inputs)
    frozen_identity = {
        role: validate_long_run_identity(
            role, case_dir, solver_log, args.start_time, args.final_time
        )
        for role, (case_dir, solver_log) in run_inputs.items()
    }
    volume_hashes = {
        region: sha256(args.volume_root / region / "V")
        for region in ("fluid", "solid1", "solid2")
    }
    for region, expected_hash in EXPECTED_VOLUME_SHA256.items():
        if volume_hashes[region] != expected_hash:
            raise ValueError(
                f"{region} volume fingerprint mismatch: expected {expected_hash}, "
                f"got {volume_hashes[region]}"
            )
    fluid_volumes = read_cell_volumes(args.volume_root / "fluid/V")
    solid1_volumes = read_cell_volumes(args.volume_root / "solid1/V")
    solid2_volumes = read_cell_volumes(args.volume_root / "solid2/V")
    _, initial_particles = read_restart(
        args.baseline / args.start_time / "gpuResidentStrictParticles.dat"
    )
    initial_ids = set(initial_particles)

    variants = {
        "baseline": case_metrics(
            args.baseline, args.final_time, fluid_volumes, solid1_volumes,
            solid2_volumes, initial_ids, args.parcel_mass, args.particle_cp,
        ),
        "baseline_repeat": case_metrics(
            args.baseline_repeat, args.final_time, fluid_volumes, solid1_volumes,
            solid2_volumes, initial_ids, args.parcel_mass, args.particle_cp,
        ),
        "candidate": case_metrics(
            args.candidate, args.final_time, fluid_volumes, solid1_volumes,
            solid2_volumes, initial_ids, args.parcel_mass, args.particle_cp,
        ),
    }
    failures: list[str] = []
    envelope: dict[str, object] = {}
    for key, reference in variants["baseline"]["signature"].items():                            
        repeat = float(variants["baseline_repeat"]["signature"][key])                       
        candidate = float(variants["candidate"]["signature"][key])                       
        passed, observed, allowed = within_envelope(float(reference), repeat, candidate)
        envelope[key] = {
            "reference": reference,
            "baseline_repeat": repeat,
            "candidate": candidate,
            "candidate_absolute_delta": observed,
            "allowed_absolute_delta": allowed,
            "pass": passed,
        }
        if not passed:
            failures.append(f"{key} exceeds baseline-repeat envelope")

    for name, metrics in variants.items():
        if (
            metrics["executed_steps"] != 128
            or metrics["delta_t"] != 1e-8
            or not math.isclose(
                float(metrics["final_time_value"]), float(EXPECTED_FINAL_TIME),
                rel_tol=0.0, abs_tol=1.0e-15,
            )
        ):
            failures.append(f"{name} does not record 128 fixed 1e-8 steps")
        particles = metrics["particles"]                            
        if particles["outlier_count"] != 0:
            failures.append(f"{name} contains particle outliers")
        state = metrics["thermal_state"]                            
        state_complete = (
            state["sha256"] == EXPECTED_THERMAL_STATE_SHA256
            and state["exchange_sequence"] == 1
            and state["completed_time_index"] == 101
            and math.isclose(
                float(state["completed_simulation_time"]), 0.10000103,
                rel_tol=0.0, abs_tol=1.0e-15,
            )
            and math.isclose(
                float(state["previous_exchange_time"]), float(EXPECTED_START_TIME),
                rel_tol=0.0, abs_tol=1.0e-15,
            )
            and float(state["particle_contact_energy_j"]) == 0.0
            and all(state["flags"].values())
        )
        if not state_complete:
            failures.append(f"{name} CHT/radiation lifecycle is incomplete")

    baseline_particles = variants["baseline"]["particles"]                            
    repeat_particles = variants["baseline_repeat"]["particles"]                            
    candidate_particles = variants["candidate"]["particles"]                            
    exact_stat_keys = (
        "status_counts", "exited", "unexpected_ids",
        "cell_occupancy", "temperature_histogram", "speed_histogram",
    )
    statistical_gates: dict[str, bool] = {}
    for key in exact_stat_keys:
        passed = candidate_particles[key] in (baseline_particles[key], repeat_particles[key])
        statistical_gates[key] = passed
        if not passed:
            failures.append(f"particle statistic {key} is outside baseline-repeat range")

    initial_metrics = case_metrics(
        args.baseline, args.start_time, fluid_volumes, solid1_volumes,
        solid2_volumes, initial_ids, args.parcel_mass, args.particle_cp,
    )
    initial_energy = float(initial_metrics["signature"]["composite_energy"])                       
    energy_changes = {
        name: float(metrics["signature"]["composite_energy"]) - initial_energy                       
        for name, metrics in variants.items()
    }
    sign_reference = math.copysign(1.0, energy_changes["baseline"])
    sign_gate = all(
        value == 0.0 or math.copysign(1.0, value) == sign_reference
        for value in energy_changes.values()
    )
    if not sign_gate:
        failures.append("composite energy-change sign reversal")

    result = {
        "gate_result": "PASS" if not failures else "FAIL",
        "frozen_identity": frozen_identity,
        "frozen_physical_parameters": {
            "parcel_mass": args.parcel_mass,
            "particle_cp": args.particle_cp,
            "start_time": args.start_time,
            "final_time": args.final_time,
        },
        "volume_hashes": volume_hashes,
        "fixed_temperature_bins": list(TEMPERATURE_BINS[:-1]) + ["inf"],
        "fixed_speed_bins": list(SPEED_BINS[:-1]) + ["inf"],
        "conservation_envelope": envelope,
        "statistical_gates": statistical_gates,
        "energy_changes_from_identical_initial_state": energy_changes,
        "energy_change_sign_gate": sign_gate,
        "variants": variants,
        "failures": failures,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
