#!/usr/bin/env python3
"""Materialize isolated solver source trees; FP32 overlays never use include precedence."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE_DIRS = {"applications", "common", "gpu", "legacy", "tool", "third_party", "validation"}
SOURCE_FILES = {"Allwmake", "env.sh", ".gitignore"}

def inputs(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    )
    paths = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        rel = Path(os.fsdecode(item))
        if rel.parts[0] in SOURCE_DIRS or str(rel) in SOURCE_FILES:
            if (root / rel).is_file() or (root / rel).is_symlink():
                paths.add(rel)
    return sorted(paths)

def copy_one(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        link = os.readlink(source)
        if target.is_symlink() and os.readlink(target) == link:
            return
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(link)
    else:
        if target.is_symlink():
            target.unlink()
        if not target.exists() or source.read_bytes() != target.read_bytes():
            shutil.copy2(source, target)

def materialize(root: Path, bits: int) -> Path:
    root = root.resolve()
    base = root / ".build" / "precision" / ("fp" + str(bits))
    destination = base / "source"
    destination.mkdir(parents=True, exist_ok=True)
    selected = {rel: root / rel for rel in inputs(root)}
    if bits == 32:
        overlay = root / "precision" / "fp32"
        if overlay.exists():
            for source in overlay.rglob("*"):
                if source.is_file() or source.is_symlink():
                    rel = source.relative_to(overlay)
                    selected[rel] = source
    manifest_path = base / "source-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for name in previous["files"]:
            rel = Path(name)
            if rel not in selected:
                stale = destination / rel
                # Manifest entries are relative names created by this tool.
                if not rel.is_absolute() and ".." not in rel.parts and (stale.is_file() or stale.is_symlink()):
                    stale.unlink()
    records = {}
    for rel, source in sorted(selected.items()):
        copy_one(source, destination / rel)
        content = os.fsencode(os.readlink(source)) if source.is_symlink() else source.read_bytes()
        records[str(rel)] = hashlib.sha256(content).hexdigest()
    build_header = destination / "common" / "PrecisionBuild.H"
    build_header.parent.mkdir(parents=True, exist_ok=True)
    build_header.write_text(
        "#pragma once\nnamespace ugkp { namespace precision {\n"
        + "inline constexpr int compiledBits = " + str(bits) + ";\n"
        + 'inline constexpr const char* policy = "' + ("legacy64-v1" if bits == 64 else "fp32-v1") + '";\n'
        + "}}\n"
    )
    records["common/PrecisionBuild.H"] = hashlib.sha256(build_header.read_bytes()).hexdigest()
    manifest = {
        "generatedFiles": ["common/PrecisionBuild.H"],
        "precision": bits,
        "sourceRoot": str(root),
        "sourceCommit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
    args = parser.parse_args()
    print(materialize(args.source_root, args.precision))

if __name__ == "__main__":
    main()
