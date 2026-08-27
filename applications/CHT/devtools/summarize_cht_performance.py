#!/usr/bin/env python3
                                                                     

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

DEVTOOLS = Path(__file__).resolve().parent
if str(DEVTOOLS) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS))

from cht_v2_contract import (
    PERFORMANCE_MIE_SHA256,
    case_contracts,
    resolve_single_mie_table,
)


KERNELS = {
    "pool": "accumulatePoissonPoolParticlesByCellKernel",
    "relax": "relaxParticlesToResidentGasKernel",
    "track": "trackParticlesLocalFaceWalkKernel",
}
VARIANTS = ("baseline", "v1", "v2")
EXPECTED_BINARY_SHA256 = {
    "baseline": "88dce4081f19700fd4b5a0ee1c33d1c9c013b3e2900cd8c7308a557f41c32b7e",
    "v1": "2ef261fb5cd469915c39f216d999cafeb63004181b21ecbd74de52f358ec2814",
    "v2": "19cdcbfa76be64da916ca1a2eb66751b03da464c5cc9543f58d3e0824b37b696",
}
EXPECTED_CASE_SHA256 = {
    "system/controlDict": "a507923f823b868bd8345d5232313bfea6a8d86220e4226fddf33b4ee70c74fb",
    "constant/solidThermalCouplingProperties": "f529024026d7cdbb8dbc98aaaae3619084db699b8715812a0cc8fad5999da6eb",
    "0.10000002/thermalExchangeState": "20de8b0670d8b958ff4551d481c1687f74d55a3a8e61984d6ccb4eaac5142328",
    "0.10000002/gpuResidentStrictParticles.dat": "680b74506411228c969466ccfa6639cd9a799ffe3212d851b60d17dc60bbf268",
}
NONFINITE = re.compile(
    r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])",
    flags=re.IGNORECASE,
)
DT_SEQUENCE_SHA256 = "f21c5e2335958c7a45fc260b9dda9da77ca95dfd621bf36ec7862473b438586d"
EXPECTED_CASE_TREE_SHA256 = "80768c2c10bfe5a359bd2f68c7b740e3d3ebac3c537a0380f0a0c84739cb071d"
EXPECTED_RAW_EVIDENCE_SHA256 = {
    "measured-profile-r1-baseline": "f2e0b0e8159f146b46a470264726570d299f4f43d30fa545dedbdcab5396af3a",
    "measured-profile-r1-v1": "909181bc84a55cc6fdb7d0311985ba42d658440568d3cb09736f07e53d760fd8",
    "measured-profile-r1-v2": "bcb1ffacb4a04656dbc5d953a98142c6686f942d05e70f0048170b9be816436d",
    "measured-profile-r2-baseline": "708ff2f7c2f5e893f2216d7313335c82286d5de89e67af2df7edb6d9fc219987",
    "measured-profile-r2-v1": "50604c2f60877a6fbd9db3ad7f01f1456603caad022d9951c6e1125a87e01ed3",
    "measured-profile-r2-v2": "ab3ecb65436d07b907eefdabe5ddd8ffbcd31b3e04c6b28b571b00b113c180d1",
    "measured-profile-r3-baseline": "a055b4c49caffe2d02d6f21cb9b8ea315b316a2da5fef342b22d02c5d2d657ed",
    "measured-profile-r3-v1": "e43110e98f59dc46f233e7a22c3fddef255a957656dd7f25721db2069db2a0a1",
    "measured-profile-r3-v2": "94fddd7291a6f03b96b1231bc34841e4612df0874fb2f2f5260e0ec13d7daef8",
    "wall-r1-baseline": "fe92d9d6525d8acaf043912a41fea0d4ba370fd9b53cd73492b279baef2f53e7",
    "wall-r1-v1": "4310bcb53100deda7a65a9772647df439657a819328e250bc8d6a6d120d66281",
    "wall-r1-v2": "e96a27839b057686989cb6760187caca40ad2d88b800b93fbad524b56422375e",
    "wall-r2-baseline": "a9598481bb07dcb908cdd38ab4dc167b569e564cf50731f7a0d387e80b2b2068",
    "wall-r2-v1": "9913e9a1f6b32294677d9b2e6a17d44e822f1a26b706a6bdeb5853797517119d",
    "wall-r2-v2": "80695595e4af67ab18c64b5de4f689022cd229ab3a75d98faa8e2b72c47ff42a",
    "wall-r3-baseline": "42d0239cf25fd15a9e17578f0b8b63fe912e745cda97936c8bbdc053e56f7c0d",
    "wall-r3-v1": "3812787400b2f2460aad3dbcbd4614fe639f7df18d8154088b81be98ca8a4b48",
    "wall-r3-v2": "411d6ad84fbe578b87ecaf135ea101c66564d5f91b52f93b9c58320ed7f26d4f",
}


@dataclass(frozen=True)
class KernelRun:
    round_id: int
    variant: str
    pool_median_ns: float
    relax_median_ns: float
    track_median_ns: float
    total_cuda_kernel_ns: float


@dataclass(frozen=True)
class WallRun:
    round_id: int
    variant: str
    wall_seconds: float
    min_temperature_c: float
    max_temperature_c: float
    max_memory_mib: float
    throttle_masks: str


def parse_kernel_csv(path: Path) -> dict[str, float]:
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            try:
                float(row[0])
                float(row[1])
                int(row[2])
            except (ValueError, IndexError):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no CUDA kernel rows in {path}")

    for row in rows:
        for index in (0, 1, 3, 4):
            value = float(row[index])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid CUDA metric {value!r} in {path}")

    result = {"total_cuda_kernel_ns": sum(float(row[1]) for row in rows)}
    if not math.isfinite(result["total_cuda_kernel_ns"]):
        raise ValueError(f"non-finite total CUDA time in {path}")
    for key, name in KERNELS.items():
        matches = [row for row in rows if name in row[-1]]
        if len(matches) != 1:
            raise ValueError(f"expected one {name} row in {path}, got {len(matches)}")
        row = matches[0]
        if int(row[2]) != 128:
            raise ValueError(f"expected 128 {name} instances in {path}, got {row[2]}")
        result[f"{key}_median_ns"] = float(row[4])
    return result


def parse_elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    if not match:
        raise ValueError(f"missing elapsed wall time in {path}")
    fields = [float(field) for field in match.group(1).split(":")]
    if not all(math.isfinite(field) and field >= 0.0 for field in fields):
        raise ValueError(f"invalid elapsed wall time in {path}")
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"invalid elapsed wall time in {path}")


def parse_telemetry(path: Path) -> tuple[float, float, float, str]:
    temperatures: list[float] = []
    memories: list[float] = []
    masks: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) < 10:
                continue
            temperatures.append(float(row[3]))
            memories.append(float(row[7]))
            masks.add(row[9])
    if not temperatures:
        raise ValueError(f"no GPU telemetry rows in {path}")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (*temperatures, *memories)
    ):
        raise ValueError(f"invalid GPU telemetry metric in {path}")
    return min(temperatures), max(temperatures), max(memories), "|".join(sorted(masks))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(
    root: Path,
    relative_roots: tuple[str, ...] = (".",),
    excluded_names: frozenset[str] = frozenset(),
) -> str:
                                                                                      
    files: list[Path] = []
    for relative_root in relative_roots:
        base = root / relative_root
        if not base.is_dir():
            raise ValueError(f"missing fingerprint directory: {base}")
        files.extend(
            path for path in base.rglob("*")
            if path.is_file() and path.name not in excluded_names
        )
    if not files:
        raise ValueError(f"empty fingerprint input: {root}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + hashlib.sha256(path.read_bytes()).digest() + b"\n")
    return digest.hexdigest()


def raw_evidence_sha256(log_dir: Path) -> str:
    return tree_sha256(log_dir, excluded_names=frozenset({"manifest-audit.txt"}))


def performance_case_sha256(case_dir: Path) -> str:
    return tree_sha256(case_dir, ("system", "constant", "0.10000002"))


def validate_raw_evidence_identity(log_dir: Path) -> str:
    expected = EXPECTED_RAW_EVIDENCE_SHA256.get(log_dir.name)
    if expected is None:
        raise ValueError(f"unrecognized retained performance evidence: {log_dir}")
    actual = raw_evidence_sha256(log_dir)
    if actual != expected:
        raise ValueError(
            f"raw performance evidence fingerprint mismatch in {log_dir}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def validate_performance_case(case_dir: Path) -> str:
    failures = case_contracts(
        case_dir,
        "fallback",
        require_fresh=False,
        require_prepared=False,
        require_performance_seed=True,
    )
    if failures:
        raise ValueError(
            f"performance case contract failed in {case_dir}: " + "; ".join(failures)
        )
    actual = performance_case_sha256(case_dir)
    if actual != EXPECTED_CASE_TREE_SHA256:
        raise ValueError(
            f"performance case snapshot mismatch in {case_dir}: "
            f"expected {EXPECTED_CASE_TREE_SHA256}, got {actual}"
        )
    return actual


def parse_manifest(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    fields: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        hash_match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if hash_match:
            hashes[str(Path(hash_match.group(2)).resolve())] = hash_match.group(1)
        elif "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields, hashes


def require_recorded_hash(
    hashes: dict[str, str], path: Path, expected_sha: str, description: str
) -> None:
    resolved = str(path.resolve())
    recorded = hashes.get(resolved)
    if recorded != expected_sha:
        raise ValueError(
            f"{description} manifest hash mismatch: expected {expected_sha}, got {recorded}"
        )
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError(
            f"{description} current-file hash mismatch: expected {expected_sha}, got {actual}"
        )


def validate_manifest(log_dir: Path, variant: str, kind: str) -> None:
    fields, hashes = parse_manifest(log_dir / "manifest.txt")
    label = fields.get("label")
    if label != log_dir.name or not label.endswith(f"-{variant}"):
        raise ValueError(f"manifest label/variant mismatch in {log_dir}: {label!r}")

    binary_text = fields.get("binary")
    case_text = fields.get("case")
    if binary_text is None or case_text is None:
        raise ValueError(f"manifest lacks binary/case identity in {log_dir}")
    binary = Path(binary_text)
    case_dir = Path(case_text)
    require_recorded_hash(
        hashes, binary, EXPECTED_BINARY_SHA256[variant], f"{variant} executable"
    )
    for relative, expected_sha in EXPECTED_CASE_SHA256.items():
        require_recorded_hash(
            hashes, case_dir / relative, expected_sha, f"performance case {relative}"
        )

    preflight = (log_dir / "preflight-contract.log").read_text(encoding="utf-8").strip()
    if preflight != "PASS: UGKP_CHT UGKP V2 contract":
        raise ValueError(f"preflight contract did not pass in {log_dir}: {preflight!r}")
    if fields.get("fatal_lines") != "0":
        raise ValueError(f"manifest reports fatal/non-finite lines in {log_dir}")
    if kind == "nsys":
        required = {
            "profile_exit_code": "0",
            "expected_steps": "128",
            "delta_t": "1e-8",
            "pool_instances": "128",
            "relax_instances": "128",
            "track_instances": "128",
            "cht_exchange_lines": "1",
        }
    elif kind == "wall":
        required = {"solver_exit_code": "0"}
    else:
        raise ValueError(f"unknown manifest kind: {kind}")
    for key, expected in required.items():
        if fields.get(key) != expected:
            raise ValueError(
                f"manifest {key} mismatch in {log_dir}: "
                f"expected {expected!r}, got {fields.get(key)!r}"
            )


def validate_audit_sidecar(
    log_dir: Path,
    variant: str,
    kind: str,
    paired_nsys_dir: Path | None = None,
) -> None:
    fields, _ = parse_manifest(log_dir / "manifest-audit.txt")
    expected_raw_evidence = EXPECTED_RAW_EVIDENCE_SHA256.get(log_dir.name)
    if expected_raw_evidence is None:
        raise ValueError(f"unrecognized retained performance evidence: {log_dir}")
    required = {
        "audit_format": "UGKP_CHT_PERFORMANCE_MANIFEST_AUDIT_V1",
        "audit_status": "PASS",
        "kind": kind,
        "variant": variant,
        "binary_sha256": EXPECTED_BINARY_SHA256[variant],
        "expected_binary_sha256": EXPECTED_BINARY_SHA256[variant],
        "expected_steps": "128",
        "executed_steps": "128",
        "dt_sequence_sha256": DT_SEQUENCE_SHA256,
        "process_exit_code": "0",
        "exact_cht_exchange_lines": "1",
        "fatal_or_nonfinite_lines": "0",
        "case_tree_sha256": EXPECTED_CASE_TREE_SHA256,
        "raw_evidence_tree_sha256": expected_raw_evidence,
        "mie_table_sha256": PERFORMANCE_MIE_SHA256,
    }
    for relative, expected_sha in EXPECTED_CASE_SHA256.items():
        required[f"case_{relative.replace('/', '_')}_sha256"] = expected_sha
    if kind == "nsys":
        required.update(
            {"pool_instances": "128", "relax_instances": "128", "track_instances": "128"}
        )
    for key, expected in required.items():
        if fields.get(key) != expected:
            raise ValueError(
                f"audit sidecar {key} mismatch in {log_dir}: "
                f"expected {expected!r}, got {fields.get(key)!r}"
            )
    if validate_raw_evidence_identity(log_dir) != fields["raw_evidence_tree_sha256"]:
        raise ValueError(f"audit sidecar raw-evidence snapshot mismatch in {log_dir}")
    manifest_fields, _ = parse_manifest(log_dir / "manifest.txt")
    case_dir = Path(manifest_fields["case"])
    if validate_performance_case(case_dir) != fields["case_tree_sha256"]:
        raise ValueError(f"audit sidecar case snapshot mismatch in {log_dir}")
    mie_table = resolve_single_mie_table(
        case_dir, case_dir / "constant/solidThermalCouplingProperties"
    )
    if sha256(mie_table) != fields["mie_table_sha256"]:
        raise ValueError(f"audit sidecar Mie-table snapshot mismatch in {log_dir}")
    for key in (
        "gpu_start_temperature_c",
        "gpu_start_sm_clock_mhz",
        "gpu_start_power_limit_raw",
        "gpu_start_throttle_mask",
        "executed_steps_basis",
    ):
        if not fields.get(key):
            raise ValueError(f"audit sidecar lacks {key} in {log_dir}")
    if fields.get("raw_manifest_sha256") != sha256(log_dir / "manifest.txt"):
        raise ValueError(f"audit sidecar no longer matches raw manifest in {log_dir}")
    solver_name = "solver-and-profiler.log" if kind == "nsys" else "solver.log"
    if fields.get("solver_log_sha256") != sha256(log_dir / solver_name):
        raise ValueError(f"audit sidecar no longer matches solver log in {log_dir}")
    runtime_name = "profiler-runtime.txt" if kind == "nsys" else "runtime.txt"
    if fields.get("runtime_log_sha256") != sha256(log_dir / runtime_name):
        raise ValueError(f"audit sidecar no longer matches runtime log in {log_dir}")
    if kind == "nsys":
        if fields.get("nsys_stats_sha256") != sha256(log_dir / "cuda-gpu-kern-sum.csv"):
            raise ValueError(f"audit sidecar no longer matches Nsight stats in {log_dir}")
    else:
        if fields.get("gpu_telemetry_sha256") != sha256(log_dir / "gpu-telemetry.csv"):
            raise ValueError(f"audit sidecar no longer matches GPU telemetry in {log_dir}")
    if kind == "wall":
        if paired_nsys_dir is None:
            raise ValueError(f"wall audit lacks paired Nsight directory: {log_dir}")
        if fields.get("paired_nsys_manifest_audit_sha256") != sha256(
            paired_nsys_dir / "manifest-audit.txt"
        ):
            raise ValueError(f"wall audit no longer matches paired Nsight manifest in {log_dir}")
        if fields.get("paired_nsys_stats_sha256") != sha256(
            paired_nsys_dir / "cuda-gpu-kern-sum.csv"
        ):
            raise ValueError(f"wall audit no longer matches paired Nsight stats in {log_dir}")


def validate_solver_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(" CHTdeltaT = ") != 1:
        raise ValueError(f"expected exactly one CHT exchange in {path}")
    if re.search(r"FOAM FATAL|CUDA error|segmentation fault", text, flags=re.IGNORECASE):
        raise ValueError(f"fatal solver text in {path}")
    if NONFINITE.search(text):
        raise ValueError(f"non-finite numeric text in {path}")


def validate_runtime_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not re.search(r"^\s*Exit status:\s*0\s*$", text, flags=re.MULTILINE):
        raise ValueError(f"runtime log does not record exit status 0: {path}")
    if not re.search(r"Elapsed \(wall clock\) time", text):
        raise ValueError(f"runtime log lacks elapsed time: {path}")


def median(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"median requires finite non-empty values: {values}")
    return float(statistics.median(values))


def percent_improvement(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or reference <= 0.0 or not math.isfinite(candidate):
        raise ValueError(
            f"performance improvement requires finite positive reference and finite candidate: "
            f"{reference!r}, {candidate!r}"
        )
    return 100.0 * (reference - candidate) / reference


def evaluate_gates(
    kernel_runs: list[KernelRun],
    kernel_summary: dict[str, dict[str, float]],
    wall_summary: dict[str, dict[str, float]],
    minimum_target_gain: float = 5.0,
    maximum_regression: float = 2.0,
) -> tuple[dict[str, float | bool], list[str]]:
                                                                                 
    v1_pool_gain = percent_improvement(
        kernel_summary["baseline"]["pool_median_ns"],
        kernel_summary["v1"]["pool_median_ns"],
    )
    v2_relax_gain = percent_improvement(
        kernel_summary["v1"]["relax_median_ns"],
        kernel_summary["v2"]["relax_median_ns"],
    )
    v2_cuda_gain = percent_improvement(
        kernel_summary["baseline"]["total_cuda_kernel_ns"],
        kernel_summary["v2"]["total_cuda_kernel_ns"],
    )
    v2_wall_gain = percent_improvement(
        wall_summary["baseline"]["median"],
        wall_summary["v2"]["median"],
    )
    baseline_pool = [run.pool_median_ns for run in kernel_runs if run.variant == "baseline"]
    v1_pool = [run.pool_median_ns for run in kernel_runs if run.variant == "v1"]
    v1_pool_ranges_separated = max(v1_pool) < min(baseline_pool)

    failures: list[str] = []
    if v1_pool_gain < minimum_target_gain:
        failures.append(
            f"V1 pool gain {v1_pool_gain:.2f}% is below {minimum_target_gain:.2f}%"
        )
    if not v1_pool_ranges_separated:
        failures.append("V1 and baseline pool per-report median ranges overlap")
    if v2_relax_gain < minimum_target_gain:
        failures.append(
            f"V2 relaxation gain {v2_relax_gain:.2f}% is below {minimum_target_gain:.2f}%"
        )
    if v2_cuda_gain < -maximum_regression:
        failures.append(
            f"V2 total CUDA regression {-v2_cuda_gain:.2f}% exceeds {maximum_regression:.2f}%"
        )
    if v2_wall_gain < -maximum_regression:
        failures.append(
            f"V2 wall regression {-v2_wall_gain:.2f}% exceeds {maximum_regression:.2f}%"
        )
    return (
        {
            "v1_pool_gain": v1_pool_gain,
            "v2_relax_gain": v2_relax_gain,
            "v2_cuda_gain": v2_cuda_gain,
            "v2_wall_gain": v2_wall_gain,
            "v1_pool_ranges_separated": v1_pool_ranges_separated,
        },
        failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-root", type=Path, required=True)
    parser.add_argument("--wall-root", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    kernel_runs: list[KernelRun] = []
    wall_runs: list[WallRun] = []
    for round_id in (1, 2, 3):
        for variant in VARIANTS:
            nsys_dir = args.nsys_root / f"measured-profile-r{round_id}-{variant}"
            validate_manifest(nsys_dir, variant, "nsys")
            validate_audit_sidecar(nsys_dir, variant, "nsys")
            validate_solver_log(nsys_dir / "solver-and-profiler.log")
            validate_runtime_log(nsys_dir / "profiler-runtime.txt")
            metrics = parse_kernel_csv(nsys_dir / "cuda-gpu-kern-sum.csv")
            kernel_runs.append(KernelRun(round_id, variant, **metrics))

            wall_dir = args.wall_root / f"wall-r{round_id}-{variant}"
            validate_manifest(wall_dir, variant, "wall")
            validate_audit_sidecar(wall_dir, variant, "wall", paired_nsys_dir=nsys_dir)
            validate_solver_log(wall_dir / "solver.log")
            validate_runtime_log(wall_dir / "runtime.txt")
            telemetry = parse_telemetry(wall_dir / "gpu-telemetry.csv")
            wall_runs.append(
                WallRun(
                    round_id,
                    variant,
                    parse_elapsed_seconds(wall_dir / "runtime.txt"),
                    *telemetry,
                )
            )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "kind", "round", "variant", "pool_median_ns", "relax_median_ns",
                "track_median_ns", "total_cuda_kernel_ns", "wall_seconds",
                "min_temperature_c", "max_temperature_c", "max_memory_mib",
                "throttle_masks",
            ]
        )
        for run in kernel_runs:
            writer.writerow(
                [
                    "nsys", run.round_id, run.variant, run.pool_median_ns,
                    run.relax_median_ns, run.track_median_ns,
                    run.total_cuda_kernel_ns, "", "", "", "", "",
                ]
            )
        for run in wall_runs:
            writer.writerow(
                [
                    "wall", run.round_id, run.variant, "", "", "", "",
                    run.wall_seconds, run.min_temperature_c, run.max_temperature_c,
                    run.max_memory_mib, run.throttle_masks,
                ]
            )

    kernel_summary: dict[str, dict[str, float]] = {}
    wall_summary: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        selected_kernels = [run for run in kernel_runs if run.variant == variant]
        selected_walls = [run for run in wall_runs if run.variant == variant]
        kernel_summary[variant] = {
            key: median([getattr(run, key) for run in selected_kernels])
            for key in (
                "pool_median_ns", "relax_median_ns", "track_median_ns",
                "total_cuda_kernel_ns",
            )
        }
        wall_values = [run.wall_seconds for run in selected_walls]
        wall_summary[variant] = {
            "median": median(wall_values),
            "min": min(wall_values),
            "max": max(wall_values),
            "min_temperature": min(run.min_temperature_c for run in selected_walls),
            "max_temperature": max(run.max_temperature_c for run in selected_walls),
            "max_memory": max(run.max_memory_mib for run in selected_walls),
        }

    gate_metrics, gate_failures = evaluate_gates(
        kernel_runs, kernel_summary, wall_summary
    )
    v1_pool_gain = float(gate_metrics["v1_pool_gain"])
    v2_relax_gain = float(gate_metrics["v2_relax_gain"])
    v2_cuda_gain = float(gate_metrics["v2_cuda_gain"])
    v2_wall_gain = float(gate_metrics["v2_wall_gain"])

    lines = [
        "# CHT UGKP V2 fixed-seed performance summary",
        "",
        "Every measured report contains 128 pool, relaxation, and tracking kernel instances and exactly one CHT exchange.",
        "",
        "| Variant | Pool median (us) | Relax median (us) | Track median (us) | Total CUDA kernels (ms) | Wall median [range] (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        k = kernel_summary[variant]
        w = wall_summary[variant]
        lines.append(
            f"| {variant} | {k['pool_median_ns']/1e3:.3f} | "
            f"{k['relax_median_ns']/1e3:.3f} | {k['track_median_ns']/1e3:.3f} | "
            f"{k['total_cuda_kernel_ns']/1e6:.3f} | "
            f"{w['median']:.2f} [{w['min']:.2f}, {w['max']:.2f}] |"
        )
    lines.extend(
        [
            "",
            f"- V1 pool-kernel median improvement vs baseline: {v1_pool_gain:.2f}% (gate >= 5%).",
            f"- V1/baseline pool per-report median ranges separated: {str(bool(gate_metrics['v1_pool_ranges_separated'])).lower()}.",
            f"- V2 relaxation-kernel median improvement vs V1: {v2_relax_gain:.2f}% (gate >= 5%).",
            f"- V2 total CUDA-kernel improvement vs baseline: {v2_cuda_gain:.2f}% (no-regression gate >= -2%).",
            f"- V2 profiler-free wall improvement vs baseline: {v2_wall_gain:.2f}% (no-regression gate >= -2%).",
            f"- Wall-run GPU temperature envelope: {min(v['min_temperature'] for v in wall_summary.values()):.0f}–{max(v['max_temperature'] for v in wall_summary.values()):.0f} C.",
            f"- Maximum observed device memory: {max(v['max_memory'] for v in wall_summary.values()):.0f} MiB of 8188 MiB.",
            "- `power.limit` remained `N/A`; raw telemetry and throttle masks are retained per run.",
            "",
            "Gate result: PASS." if not gate_failures else "Gate result: FAIL.",
            *(f"- {failure}" for failure in gate_failures),
            "",
            (
                "Decision: retain both V1 and V2. Both target-kernel gates pass, and final V2 does not regress total CUDA or whole-process wall time."
                if not gate_failures
                else "Decision: reject the measured candidate; one or more frozen performance gates failed."
            ),
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), end="")
    return 1 if gate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
