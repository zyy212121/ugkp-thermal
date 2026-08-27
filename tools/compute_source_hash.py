#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


SOURCE_SUFFIXES = {
    ".C",
    ".H",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".inl",
}

SOURCE_ROOTS = (
    "applications/gasUGKP",
    "applications/FSH",
    "applications/CHT",
    "common",
    "gpu",
)

EXPLICIT_INPUTS = (
    "Allwmake",
    "tools/compute_source_hash.py",
    "applications/gasUGKP/private_backend/build_private_backend.sh",
    "applications/FSH/private_backend/build_private_backend.sh",
    "applications/CHT/tools/build_cuda_solver.sh",
)

EXCLUDED_PARTS = {
    ".pytest_cache",
    "__pycache__",
    "build_logs",
    "devtools",
    "docs",
    "examples",
    "legacy",
    "postProcessing",
    "results",
    "tests",
}


def _is_build_input(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.suffix in SOURCE_SUFFIXES:
        return True
    if relative.name == "Makefile":
        return True
    return relative.name in {"files", "options"} and relative.parent.name == "Make"


def build_input_paths(root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for relative_text in EXPLICIT_INPUTS:
        relative = Path(relative_text)
        if os.path.lexists(root / relative):
            paths.add(relative)
    for source_root_text in SOURCE_ROOTS:
        source_root = root / source_root_text
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(root)
            if _is_build_input(relative):
                paths.add(relative)
    return tuple(sorted(paths, key=lambda path: path.as_posix().encode()))


def _field(hasher: hashlib._Hash, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def source_manifest(root: Path) -> tuple[tuple[str, str, str, str], ...]:
    entries = []
    for relative in build_input_paths(root):
        path = root / relative
        relative_text = relative.as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve(strict=True)
            content_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            entries.append((relative_text, "link", target, content_hash))
        else:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((relative_text, "file", "", content_hash))
    return tuple(entries)


def source_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    for entry in source_manifest(root):
        for value in entry:
            _field(hasher, value.encode())
    return hasher.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 2:
        root = Path(sys.argv[1]).resolve()
    elif len(sys.argv) != 1:
        return 2
    print(source_hash(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
