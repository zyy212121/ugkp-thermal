#!/usr/bin/env python3
                                                                                  

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for manifest in sorted(args.log_root.rglob("manifest.txt")):
        log_dir = manifest.parent
        solver = log_dir / "solver.log"
        if not solver.is_file():
            solver = log_dir / "solver-and-profiler.log"
        if not solver.is_file():
            continue
        fields = manifest_fields(manifest)
        exact = solver.read_text(encoding="utf-8").count(" CHTdeltaT = ")
        recorded = fields.get("cht_exchange_lines", "")
        rows.append(
            [
                str(log_dir.relative_to(args.log_root)),
                fields.get("label", ""),
                recorded,
                exact,
                "MATCH" if recorded == str(exact) else "LEGACY_COUNT_CORRECTED",
                sha256(manifest),
                sha256(solver),
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "log_directory",
                "label",
                "legacy_manifest_cht_exchange_lines",
                "exact_cht_delta_t_lines",
                "audit_status",
                "raw_manifest_sha256",
                "solver_log_sha256",
            ]
        )
        writer.writerows(rows)
    corrected = sum(row[4] == "LEGACY_COUNT_CORRECTED" for row in rows)
    print(f"audited {len(rows)} manifests; corrected metadata for {corrected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
