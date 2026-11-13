#!/usr/bin/env python3
"""Report FP64/FP32 source pairs for review; never admit or reject a solver run."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def report(root):
    provenance = json.loads((root / "precision/fp32-import.json").read_text())
    original = provenance.get("base_file_sha256", {})
    imported = provenance.get("files", {})
    pairs = []
    for source in sorted((root / "precision/fp32").rglob("*")):
        if not source.is_file():
            continue
        name = str(source.relative_to(root / "precision/fp32"))
        base = original.get(name)
        current = digest(root / name)
        overlay = digest(source)
        pairs.append({
            "path": name,
            "baseline64_sha256": base,
            "current64_sha256": current,
            "current32_sha256": overlay,
            "imported32_sha256": imported.get(name),
            "root_changed_since_baseline": current != base,
            "overlay_changed_since_import": overlay != imported.get(name),
            "paired_review_requested": current != base,
        })
    return {
        "kind": "advisory_dual_precision_source_pairs",
        "baseline64_commit": provenance.get("base_fp64_commit"),
        "purpose": "Review algorithm, scheduling, parameters and functionality in both precisions. Hashes do not prove equivalence.",
        "pairs": pairs,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(report(args.source_root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(str(args.output))
    else:
        print(text, end="")

if __name__ == "__main__":
    main()
