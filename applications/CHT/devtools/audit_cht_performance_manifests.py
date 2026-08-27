#!/usr/bin/env python3
                                                                              

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

DEVTOOLS = Path(__file__).resolve().parent
if str(DEVTOOLS) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS))

from cht_v2_contract import (
    PERFORMANCE_MIE_SHA256,
    parse_single_scalar,
    parse_single_word,
    resolve_single_mie_table,
)
from summarize_cht_performance import (
    EXPECTED_BINARY_SHA256,
    EXPECTED_CASE_SHA256,
    NONFINITE,
    VARIANTS,
    parse_kernel_csv,
    parse_manifest,
    sha256,
    validate_performance_case,
    validate_raw_evidence_identity,
    validate_manifest,
    validate_runtime_log,
    validate_solver_log,
)


DT_SEQUENCE_SHA256 = "f21c5e2335958c7a45fc260b9dda9da77ca95dfd621bf36ec7862473b438586d"


def first_wall_telemetry(path: Path) -> tuple[str, str, str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) >= 10:
                return row[3], row[4], row[6], row[9]
    raise ValueError(f"no usable telemetry row in {path}")


def first_nsys_telemetry(value: str) -> tuple[str, str, str, str]:
    row = next(csv.reader([value], skipinitialspace=True))
    if len(row) < 9:
        raise ValueError(f"invalid gpu_before manifest row: {value!r}")
    return row[2], row[3], row[5], row[8]


def exact_log_counts(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    exchanges = text.count(" CHTdeltaT = ")
    fatal = int(
        bool(re.search(r"FOAM FATAL|CUDA error|segmentation fault", text, re.IGNORECASE))
        or bool(NONFINITE.search(text))
    )
    return exchanges, fatal


def schedule(case_dir: Path) -> tuple[float, float, float, int]:
    control = case_dir / "system/controlDict"
    start_from = parse_single_word(control, "startFrom")
    start = parse_single_scalar(control, "startTime")
    end = parse_single_scalar(control, "endTime")
    delta_t = parse_single_scalar(control, "deltaT")
    if start_from != "startTime":
        raise ValueError(f"performance case must use startFrom startTime: {case_dir}")
    steps_float = (end - start) / delta_t
    steps = round(steps_float)
    if abs(steps_float - steps) > 1e-9:
        raise ValueError(f"non-integral performance schedule in {case_dir}: {steps_float}")
    return start, end, delta_t, steps


def audit_one(
    log_dir: Path,
    variant: str,
    kind: str,
    paired_nsys_dir: Path | None = None,
) -> None:
    raw_evidence_hash = validate_raw_evidence_identity(log_dir)
    validate_manifest(log_dir, variant, kind)
    fields, _ = parse_manifest(log_dir / "manifest.txt")
    case_dir = Path(fields["case"])
    case_tree_hash = validate_performance_case(case_dir)
    mie_table = resolve_single_mie_table(
        case_dir, case_dir / "constant/solidThermalCouplingProperties"
    )
    mie_table_hash = sha256(mie_table)
    if mie_table_hash != PERFORMANCE_MIE_SHA256:
        raise ValueError(f"performance Mie-table fingerprint mismatch: {mie_table}")
    start, end, delta_t, expected_steps = schedule(case_dir)
    if (start, end, delta_t, expected_steps) != (0.10000002, 0.10000130, 1e-8, 128):
        raise ValueError(f"unexpected frozen schedule in {case_dir}")

    if kind == "nsys":
        solver_log = log_dir / "solver-and-profiler.log"
        runtime_log = log_dir / "profiler-runtime.txt"
        telemetry = first_nsys_telemetry(fields["gpu_before"])
        kernel_metrics = parse_kernel_csv(log_dir / "cuda-gpu-kern-sum.csv")
        del kernel_metrics
        exit_code = fields["profile_exit_code"]
        executed_basis = "128 pool/relax/track instances in the Nsight report"
    else:
        solver_log = log_dir / "solver.log"
        runtime_log = log_dir / "runtime.txt"
        telemetry_path = log_dir / "gpu-telemetry.csv"
        telemetry = first_wall_telemetry(telemetry_path)
        exit_code = fields["solver_exit_code"]
        executed_basis = (
            "fixed 128-step control hash + solver exit 0 + paired variant Nsight 128-instance report"
        )
    validate_solver_log(solver_log)
    validate_runtime_log(runtime_log)
    exchanges, fatal = exact_log_counts(solver_log)
    if exit_code != "0" or exchanges != 1 or fatal != 0:
        raise ValueError(f"invalid retained performance run in {log_dir}")

    binary = Path(fields["binary"])
    lines = [
        "audit_format=UGKP_CHT_PERFORMANCE_MANIFEST_AUDIT_V1",
        "audit_status=PASS",
        f"kind={kind}",
        f"round={log_dir.name.split('-r', 1)[1].split('-', 1)[0]}",
        f"variant={variant}",
        f"label={fields['label']}",
        f"binary={binary}",
        f"binary_sha256={sha256(binary)}",
        f"expected_binary_sha256={EXPECTED_BINARY_SHA256[variant]}",
        f"case={case_dir}",
        f"start_time={start:.17g}",
        f"end_time={end:.17g}",
        f"delta_t={delta_t:.17g}",
        f"expected_steps={expected_steps}",
        "executed_steps=128",
        f"executed_steps_basis={executed_basis}",
        f"dt_sequence_sha256={DT_SEQUENCE_SHA256}",
        f"gpu_start_temperature_c={telemetry[0]}",
        f"gpu_start_sm_clock_mhz={telemetry[1]}",
        f"gpu_start_power_limit_raw={telemetry[2]}",
        f"gpu_start_throttle_mask={telemetry[3]}",
        f"process_exit_code={exit_code}",
        f"exact_cht_exchange_lines={exchanges}",
        f"fatal_or_nonfinite_lines={fatal}",
        f"case_tree_sha256={case_tree_hash}",
        f"raw_evidence_tree_sha256={raw_evidence_hash}",
        f"mie_table_sha256={mie_table_hash}",
        f"raw_manifest_sha256={sha256(log_dir / 'manifest.txt')}",
        f"solver_log_sha256={sha256(solver_log)}",
        f"runtime_log_sha256={sha256(runtime_log)}",
    ]
    for relative, expected_sha in EXPECTED_CASE_SHA256.items():
        lines.append(f"case_{relative.replace('/', '_')}_sha256={expected_sha}")
    if kind == "wall":
        lines.append(f"gpu_telemetry_sha256={sha256(log_dir / 'gpu-telemetry.csv')}")
        if paired_nsys_dir is None:
            raise ValueError(f"wall audit requires paired Nsight evidence: {log_dir}")
        lines.append(
            "paired_nsys_manifest_audit_sha256="
            f"{sha256(paired_nsys_dir / 'manifest-audit.txt')}"
        )
        lines.append(
            "paired_nsys_stats_sha256="
            f"{sha256(paired_nsys_dir / 'cuda-gpu-kern-sum.csv')}"
        )
    else:
        lines.extend(
            [
                "pool_instances=128",
                "relax_instances=128",
                "track_instances=128",
                f"nsys_stats_sha256={sha256(log_dir / 'cuda-gpu-kern-sum.csv')}",
            ]
        )
    (log_dir / "manifest-audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-root", type=Path, required=True)
    parser.add_argument("--wall-root", type=Path, required=True)
    args = parser.parse_args()
    for round_id in (1, 2, 3):
        for variant in VARIANTS:
            nsys_dir = args.nsys_root / f"measured-profile-r{round_id}-{variant}"
            audit_one(
                nsys_dir,
                variant,
                "nsys",
            )
            audit_one(
                args.wall_root / f"wall-r{round_id}-{variant}",
                variant,
                "wall",
                paired_nsys_dir=nsys_dir,
            )
    print("PASS: audited 18 immutable performance manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
