#!/usr/bin/env python3
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




GENERATED_DIRS = {
    ".build", "__pycache__", ".pytest_cache", ".validation-build",
    ".validation-bin", ".generated", "VTK", "build_logs", "run_logs",
}
GENERATED_PATHS = {
    "validation/composition_transport/evidence",
    "validation/shared_boundary/evidence",
    "validation/composition_transport/REVIEW.txt",
    "validation/composition_transport/PR_WALL_REVIEW.txt",
    "validation/shared_boundary/REVIEW.txt",
}

def generated(rel: Path, directory: bool = False) -> bool:
    parts = rel.parts
    directories = parts if directory else parts[:-1]
    if any(part in GENERATED_DIRS or part.startswith("processor") for part in directories):
        return True
    for index, part in enumerate(directories):
        if index and ((parts[index - 1] == "Make" and part.startswith("linux"))
                      or (parts[index - 1] == "private_backend" and part == "build")):
            return True
    name = rel.name
    if not directory:
        if (name == "core" or name.startswith(("core.", "log."))
                or name.endswith((".pyc", ".log", ".tmp", ".foam", ".a", ".o", ".so", ":Zone.Identifier"))):
            return True
        if rel.as_posix().endswith("assets/radiation/alumina_mieTable.dat"):
            return True
    return any(rel.as_posix() == path or rel.as_posix().startswith(path + "/")
               for path in GENERATED_PATHS)

def archive_inputs(root: Path) -> list[Path]:
    paths = set()
    for name in sorted(SOURCE_FILES):
        source = root / name
        if source.is_file() or source.is_symlink():
            paths.add(Path(name))
    for name in sorted(SOURCE_DIRS):
        source = root / name
        if source.is_symlink():
            paths.add(Path(name))
            continue
        if not source.is_dir():
            continue
        for current, directories, files in os.walk(source, followlinks=False):
            current = Path(current)
            kept = []
            for directory in sorted(directories):
                child = current / directory
                rel = child.relative_to(root)
                if generated(rel, directory=True):
                    continue
                if child.is_symlink():
                    paths.add(rel)
                else:
                    kept.append(directory)
            directories[:] = kept
            for name in sorted(files):
                child = current / name
                rel = child.relative_to(root)
                if not generated(rel):
                    paths.add(rel)
    return sorted(paths)

def git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )

def source_identity(root: Path) -> tuple[str, str | None]:
    marker = root / ".git"


    if not (marker.exists() or marker.is_symlink()):
        return "archive", None
    top = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel").stdout).strip()).resolve()
    if top != root:
        return "archive", None
    head = git(root, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False)
    if head.returncode == 0:
        return "git", os.fsdecode(head.stdout).strip()


    branch = git(root, "symbolic-ref", "--quiet", "HEAD", check=False)
    if head.returncode == 1 and branch.returncode == 0:
        ref = git(root, "show-ref", "--verify", "--quiet",
                  os.fsdecode(branch.stdout).strip(), check=False)
        if ref.returncode == 1:
            return "git", None
    raise subprocess.CalledProcessError(
        head.returncode, head.args, output=head.stdout, stderr=head.stderr)

def inputs(root: Path, source_kind: str | None = None) -> list[Path]:
    if source_kind is None:
        source_kind, _ = source_identity(root.resolve())
    if source_kind == "archive":
        return archive_inputs(root)
    raw = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout
    paths = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        rel = Path(os.fsdecode(item))
        if rel.parts[0] in SOURCE_DIRS or str(rel) in SOURCE_FILES:
            if (root / rel).is_file() or (root / rel).is_symlink():
                paths.add(rel)
    return sorted(paths)

def overlay_inputs(root: Path, source_kind: str) -> list[Path]:
    overlay = root / "precision" / "fp32"
    if source_kind != "git":
        return archive_inputs(overlay)
    prefix = Path("precision/fp32")
    tracked = set(git(root, "ls-files", "-z", "--cached", "--", str(prefix)).stdout.split(b"\0"))
    raw = git(root, "ls-files", "-z", "--cached", "--others",
              "--exclude-standard", "--", str(prefix)).stdout
    paths = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        rel = Path(os.fsdecode(item)).relative_to(prefix)
        source = overlay / rel
        if rel.parts[0] not in SOURCE_DIRS and str(rel) not in SOURCE_FILES:
            continue


        if item not in tracked and generated(rel, directory=source.is_dir()):
            continue
        if source.is_file() or source.is_symlink():
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
    source_kind, source_commit = source_identity(root)
    base = root / ".build" / "precision" / ("fp" + str(bits))
    destination = base / "source"
    destination.mkdir(parents=True, exist_ok=True)
    selected = {rel: root / rel for rel in inputs(root, source_kind)}
    if bits == 32:
        overlay = root / "precision" / "fp32"
        if overlay.exists():
            for rel in overlay_inputs(root, source_kind):
                selected[rel] = overlay / rel
    manifest_path = base / "source-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for name in previous["files"]:
            rel = Path(name)
            if rel not in selected:
                stale = destination / rel

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
        "sourceKind": source_kind,
        "sourceCommit": source_commit,
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
