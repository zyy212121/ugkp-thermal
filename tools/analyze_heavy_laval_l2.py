#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import math
import statistics
import subprocess
from pathlib import Path


FACTORS = (1, 2, 4, 8, 16, 32, 64)
FIELDS = ("total_ms", "collision_pool_ms", "moments_ms")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def median(group: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in group)


def summarize(result_root: Path, archive_root: Path, output: Path) -> None:
    current = rows(result_root / "process_medians.csv")
    archived = rows(archive_root / "weighted_tail_s1_s2" / "authoritative_summaries" / "process_medians.csv")
    records = []
    for factor in FACTORS:
        l2 = [row for row in current if int(row["factor"]) == factor and int(row["measured"]) == 1]
        s1 = [row for row in archived if int(row["factor"]) == factor and row["state"] == "S1"]
        s2 = [row for row in archived if int(row["factor"]) == factor and row["state"] == "S2"]
        if len(l2) != 5 or len(s1) != 5 or len(s2) != 5:
            raise ValueError(f"k{factor}: expected five measured processes per state")
        legacy_hashes = {row["restart_sha256"] for row in s1 + s2}
        current_hashes = {row["restart_sha256"] for row in l2}
        manifest = json.loads((result_root / "inputs" / f"k{factor}" / "unified_l2_manifest.json").read_text(encoding="utf-8"))
        if legacy_hashes != {manifest["archived_restart_sha256"]} or current_hashes != {manifest["restart_sha256"]}:
            raise ValueError(f"k{factor}: restart provenance mismatch")
        record: dict[str, object] = {
            "factor": factor,
            "target_cell_initial_parcels": int(manifest["selected_source_particles"]) * factor,
            "initial_total_parcels": int(manifest["particles"]),
            "archived_restart_sha256": manifest["archived_restart_sha256"],
            "unified_restart_sha256": manifest["restart_sha256"],
        }
        for field in FIELDS:
            stem = field.removesuffix("_ms")
            for prefix, group in (("s1", s1), ("legacy_s2", s2), ("unified_l2", l2)):
                values = [float(row[field]) for row in group]
                record[f"{prefix}_{stem}_ms"] = statistics.median(values)
                record[f"{prefix}_{stem}_mean_ms"] = statistics.mean(values)
                record[f"{prefix}_{stem}_std_ms"] = statistics.stdev(values)
        record["l2_speedup_over_s1"] = float(record["s1_total_ms"]) / float(record["unified_l2_total_ms"])
        record["l2_speedup_over_legacy_s2"] = float(record["legacy_s2_total_ms"]) / float(record["unified_l2_total_ms"])
        l2_mean = float(record["unified_l2_total_mean_ms"])
        l2_cv = float(record["unified_l2_total_std_ms"]) / l2_mean
        for reference in ("s1", "legacy_s2"):
            speedup = float(record[f"l2_speedup_over_{reference}"])
            reference_mean = float(record[f"{reference}_total_mean_ms"])
            reference_cv = float(record[f"{reference}_total_std_ms"]) / reference_mean
            record[f"l2_speedup_over_{reference}_std"] = speedup * math.sqrt(reference_cv**2 + l2_cv**2)
        records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "heavy_laval_s1_s2_l2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    source_plot = Path(__file__).with_name("plot_heavy_laval_jcp.py")
    plot_path = output / "heavy_laval_s1_s2_l2.py"
    shutil.copy2(source_plot, plot_path)
    subprocess.run(["python3", str(plot_path)], cwd=output, check=True)
    manifest = {
        "schema": "unified-heavy-laval-s1-s2-l2-analysis-v1",
        "current_result_root": str(result_root),
        "archived_evidence_root": str(archive_root),
        "measured_processes_per_state_and_level": 5,
        "warmup_processes_excluded": True,
        "factors": list(FACTORS),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.result_root.resolve(), args.archive_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
