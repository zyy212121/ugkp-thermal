#!/usr/bin/env python3
                                                                                

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


CONTINUOUS = ("x", "y", "z", "ux", "uy", "uz", "temperature", "theta", "diameter")
DISCRETE = ("cell", "status", "rng")


def read_restart(path: Path) -> tuple[list[int], dict[int, dict[str, float | int]]]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2 or header[0] != "UGKP_PARTICLES_SCHEMA4":
            raise ValueError(f"expected UGKP_PARTICLES_SCHEMA4 header in {path}")
        expected = int(header[1])
        order: list[int] = []
        particles: dict[int, dict[str, float | int]] = {}
        for line_number, line in enumerate(stream, start=2):
            if not line.strip():
                continue
            words = line.split()
            if len(words) != 13:
                raise ValueError(f"{path}:{line_number}: expected 13 columns, got {len(words)}")
            values: dict[str, float | int] = {
                "x": float(words[0]), "y": float(words[1]), "z": float(words[2]),
                "ux": float(words[3]), "uy": float(words[4]), "uz": float(words[5]),
                "temperature": float(words[6]), "theta": float(words[7]),
                "diameter": float(words[8]), "cell": int(words[9]),
                "status": int(words[10]), "rng": int(words[11]),
                "orig_id": int(words[12]),
            }
            orig_id = int(values["orig_id"])
            if orig_id in particles:
                raise ValueError(f"duplicate orig_id {orig_id} in {path}")
            order.append(orig_id)
            particles[orig_id] = values
    if len(particles) != expected:
        raise ValueError(f"{path}: header count {expected}, parsed {len(particles)}")
    return order, particles


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    ref_order, ref = read_restart(reference)
    cand_order, cand = read_restart(candidate)
    ref_ids = set(ref)
    cand_ids = set(cand)
    common = sorted(ref_ids & cand_ids)

    discrete_mismatches = {name: 0 for name in DISCRETE}
    continuous: dict[str, dict[str, float | int | bool]] = {}
    for name in CONTINUOUS:
        max_abs = 0.0
        max_scaled = 0.0
        mismatch_count = 0
        for orig_id in common:
            left = float(ref[orig_id][name])
            right = float(cand[orig_id][name])
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"non-finite {name} for orig_id {orig_id}")
            delta = abs(left - right)
            max_abs = max(max_abs, delta)
            max_scaled = max(max_scaled, delta/max(1.0, abs(left), abs(right)))
            mismatch_count += int(left != right)
        continuous[name] = {
            "exact": mismatch_count == 0,
            "mismatch_count": mismatch_count,
            "max_abs": max_abs,
            "max_scaled": max_scaled,
        }

    for orig_id in common:
        for name in DISCRETE:
            discrete_mismatches[name] += int(ref[orig_id][name] != cand[orig_id][name])

    ref_cells = Counter(int(values["cell"]) for values in ref.values())
    cand_cells = Counter(int(values["cell"]) for values in cand.values())
    cell_keys = set(ref_cells) | set(cand_cells)
    occupancy_l1 = sum(abs(ref_cells[key] - cand_cells[key]) for key in cell_keys)

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_count": len(ref),
        "candidate_count": len(cand),
        "id_sets_equal": ref_ids == cand_ids,
        "missing_ids": len(ref_ids - cand_ids),
        "extra_ids": len(cand_ids - ref_ids),
        "file_order_equal": ref_order == cand_order,
        "discrete_mismatches": discrete_mismatches,
        "discrete_exact": ref_ids == cand_ids and all(value == 0 for value in discrete_mismatches.values()),
        "cell_occupancy_l1": occupancy_l1,
        "continuous": continuous,
        "continuous_all_exact": all(bool(stats["exact"]) for stats in continuous.values()),
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
    if not bool(result["id_sets_equal"]):
        failures.append(
            f"particle ID sets differ: {result['missing_ids']} missing, "
            f"{result['extra_ids']} extra"
        )
    if not bool(result["discrete_exact"]):
        failures.append(f"particle discrete state differs: {result['discrete_mismatches']}")
    if int(result["cell_occupancy_l1"]) != 0:
        failures.append(f"cell occupancy L1 is {result['cell_occupancy_l1']}, expected 0")

    if gate == "strict" and not bool(result["continuous_all_exact"]):
        failures.append("continuous particle state is not exact")
    elif gate == "tolerance":
        observed = max(
            float(stats["max_scaled"])
            for stats in result["continuous"].values()
        )
        if observed > float(max_scaled):
            failures.append(
                f"maximum scaled continuous difference {observed:.17g} "
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
        choices=("strict", "discrete", "tolerance", "report"),
        default="strict",
        help="acceptance tier; strict is the fail-safe default",
    )
    parser.add_argument(
        "--max-scaled",
        type=float,
        help="maximum scaled continuous difference for --gate tolerance",
    )
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.summary_only:
        summary = {
            "reference_count": result["reference_count"],
            "candidate_count": result["candidate_count"],
            "id_sets_equal": result["id_sets_equal"],
            "file_order_equal": result["file_order_equal"],
            "discrete_exact": result["discrete_exact"],
            "cell_occupancy_l1": result["cell_occupancy_l1"],
            "continuous_all_exact": result["continuous_all_exact"],
            "max_continuous_scaled": max(
                float(stats["max_scaled"])
                for stats in result["continuous"].values()
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(encoded, end="")
    failures = gate_failures(result, args.gate, args.max_scaled)
    if failures:
        print("particle restart acceptance failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
