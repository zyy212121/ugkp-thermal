#!/usr/bin/env python3
                                                                         

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
EXCLUDED = {
    "gpuResidentStrictParticles.dat",
    "thermalExchangeState",
    "thermalExchangeManifest",
    "gpuResidentEpsGPrev.dat",
    "gpuResidentSourceResidual.dat",
}


def numeric_tokens(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    return [float(token) for token in NUMBER.findall(text)]


def files_under(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED
    }


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    ref_files = files_under(reference)
    cand_files = files_under(candidate)
    relative_paths = sorted(set(ref_files) | set(cand_files))
    results: dict[str, object] = {}
    numeric_gate = True

    for relative in relative_paths:
        ref_path = ref_files.get(relative)
        cand_path = cand_files.get(relative)
        if ref_path is None or cand_path is None:
            results[str(relative)] = {
                "present_in_reference": ref_path is not None,
                "present_in_candidate": cand_path is not None,
            }
            numeric_gate = False
            continue
        left = numeric_tokens(ref_path)
        right = numeric_tokens(cand_path)
        if len(left) != len(right):
            results[str(relative)] = {
                "reference_numeric_count": len(left),
                "candidate_numeric_count": len(right),
                "numeric_count_equal": False,
            }
            numeric_gate = False
            continue
        max_abs = 0.0
        max_scaled = 0.0
        mismatches = 0
        for a, b in zip(left, right):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"non-finite numeric token in {relative}")
            delta = abs(a - b)
            max_abs = max(max_abs, delta)
            max_scaled = max(max_scaled, delta/max(1.0, abs(a), abs(b)))
            mismatches += int(a != b)
        results[str(relative)] = {
            "byte_exact": ref_path.read_bytes() == cand_path.read_bytes(),
            "numeric_count": len(left),
            "numeric_exact": mismatches == 0,
            "numeric_mismatch_count": mismatches,
            "max_abs": max_abs,
            "max_scaled": max_scaled,
        }

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "file_sets_equal": set(ref_files) == set(cand_files),
        "all_numeric_exact": numeric_gate and all(
            bool(stats.get("numeric_exact", False)) for stats in results.values()
        ),
        "max_abs": max(
            (float(stats.get("max_abs", 0.0)) for stats in results.values()),
            default=0.0,
        ),
        "max_scaled": max(
            (float(stats.get("max_scaled", 0.0)) for stats in results.values()),
            default=0.0,
        ),
        "files": results,
    }


def gate_failures(
    result: dict[str, object],
    gate: str,
    max_scaled: float | None = None,
) -> list[str]:
                                                                              
    if gate == "report":
        return []
    if gate == "tolerance" and max_scaled is None:
        raise ValueError("--gate tolerance requires --max-scaled")
    if max_scaled is not None and (not math.isfinite(max_scaled) or max_scaled < 0.0):
        raise ValueError("--max-scaled must be a finite non-negative value")

    failures: list[str] = []
    if not bool(result["file_sets_equal"]):
        failures.append("time-directory file sets differ")
    if gate == "strict" and not bool(result["all_numeric_exact"]):
        failures.append("numeric field content is not exact")
    elif gate == "tolerance" and float(result["max_scaled"]) > float(max_scaled):
        failures.append(
            f"maximum scaled field difference {float(result['max_scaled']):.17g} "
            f"exceeds {float(max_scaled):.17g}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--gate",
        choices=("strict", "tolerance", "report"),
        default="strict",
        help="acceptance tier; strict is the fail-safe default",
    )
    parser.add_argument(
        "--max-scaled",
        type=float,
        help="maximum scaled numeric difference for --gate tolerance",
    )
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.summary_only:
        summary = {
            "all_numeric_exact": result["all_numeric_exact"],
            "max_abs": result["max_abs"],
            "max_scaled": result["max_scaled"],
            "changed": [
                path for path, stats in result["files"].items()
                if not bool(stats.get("numeric_exact", False))
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(encoded, end="")
    failures = gate_failures(result, args.gate, args.max_scaled)
    if failures:
        print("time-directory acceptance failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
