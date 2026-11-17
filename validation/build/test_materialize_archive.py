#!/usr/bin/env python3
"""Small-tree tests for Git and standalone-archive precision materialization."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tool/precision/materialize_sources.py"
spec = importlib.util.spec_from_file_location("archive_materializer", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
IGNORE = (ROOT / ".gitignore").read_text()

def put(root, name, content="source\n"):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target

def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("common/Base.H", "common/Choice.H", "common/Fields.inc",
                 "common/remove.H", "applications/CHT/Make/files",
                 "applications/CHMT/core/Types.H", "applications/CHMT/core/Configuration.H",
                 "applications/CHMT/core/HostState.H",
                 "precision/fp32/applications/CHMT/core/Types.H",
                 "precision/fp32/applications/CHMT/core/Configuration.H",
                 "precision/fp32/applications/CHMT/core/HostState.H",
                 "tool/precision/config.json", "validation/build/check_local.py",
                 "validation/build/evidence/historical.log.txt",
                 "precision/fp32/common/Choice.H", "precision/fp32/common/Only32.inc",
                 "precision/fp32/validation/build/check_fp32.py"):
        put(root, name, name + "\n")
    put(root, ".gitignore", IGNORE)
    put(root, "Allwmake", "#!/bin/sh\n")
    put(root, "env.sh", "#!/bin/sh\n")

def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True)

def repository(root, commit=True):
    git(root, "init", "-q")
    git(root, "config", "user.name", "Materializer Test")
    git(root, "config", "user.email", "materializer-test@example.invalid")
    git(root, "add", ".")
    # The real checkout already tracks these source directories despite its
    # historical core-dump ignore pattern matching the name "core".
    git(root, "add", "-f", "applications/CHMT/core", "precision/fp32/applications/CHMT/core")
    if commit:
        git(root, "commit", "-qm", "fixture")

def manifest(root, bits):
    module.materialize(root, bits)
    return json.loads((root / ".build/precision" / ("fp" + str(bits)) /
                       "source-manifest.json").read_text())

class ArchiveMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ugkp-materialize-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)

    def test_git_archive_same_file_and_hash_manifest(self):
        checkout, archive = self.home / "checkout", self.home / "archive"
        fixture(checkout)
        shutil.copytree(checkout, archive)
        repository(checkout)
        for bits in (32, 64):
            a, b = manifest(checkout, bits), manifest(archive, bits)
            self.assertEqual(a["files"], b["files"])
            self.assertEqual(a["sourceKind"], "git")
            self.assertEqual(a["sourceCommit"], git(checkout, "rev-parse", "HEAD").stdout.strip())
            self.assertEqual(b["sourceKind"], "archive")
            self.assertIsNone(b["sourceCommit"])
            digest = hashlib.sha256(json.dumps(a["files"], sort_keys=True).encode()).hexdigest()
            print("MANIFEST", bits, len(a["files"]), digest)

    def test_archive_without_git_executable(self):
        root = self.home / "archive"
        fixture(root)
        env = dict(os.environ, PATH="")
        subprocess.run([sys.executable, str(SCRIPT), "--source-root", str(root),
                        "--precision", "32"], check=True, env=env, capture_output=True)
        with mock.patch.object(module.subprocess, "run", side_effect=AssertionError("Git called")):
            data = manifest(root, 64)
        self.assertEqual(data["sourceKind"], "archive")

    def test_archive_inside_parent_git_does_not_discover_parent(self):
        parent = self.home / "parent"
        fixture(parent)
        repository(parent)
        root = parent / "download"
        fixture(root)
        with mock.patch.object(module.subprocess, "run", side_effect=AssertionError("parent Git used")):
            data = manifest(root, 32)
        self.assertEqual(data["sourceKind"], "archive")
        self.assertIsNone(data["sourceCommit"])

    def test_different_git_top_level_is_archive_not_rejected(self):
        root = self.home / "checkout"
        fixture(root)
        repository(root)
        other = self.home / "other"
        other.mkdir()
        with mock.patch.dict(os.environ, {"GIT_WORK_TREE": str(other)}):
            data = manifest(root, 32)
        self.assertEqual(data["sourceKind"], "archive")
        self.assertIsNone(data["sourceCommit"])

    def test_unborn_git_preserves_git_selection(self):
        root = self.home / "unborn"
        fixture(root)
        repository(root, commit=False)
        data = manifest(root, 64)
        self.assertEqual(data["sourceKind"], "git")
        self.assertIsNone(data["sourceCommit"])
        self.assertIn("common/Fields.inc", data["files"])

    def test_declared_git_failures_are_not_silently_archives(self):
        root = self.home / "broken"
        fixture(root)
        (root / ".git").mkdir()
        with self.assertRaises(subprocess.CalledProcessError):
            module.materialize(root, 64)
        with mock.patch.object(module.subprocess, "run", side_effect=FileNotFoundError("git")):
            with self.assertRaises(FileNotFoundError):
                module.materialize(root, 64)
        self.assertFalse((root / ".build").exists())

    def test_git_ls_files_failure_is_not_archive_fallback(self):
        root = self.home / "git"
        fixture(root)
        repository(root)
        original = module.subprocess.run
        def failing(command, **kw):
            if "ls-files" in command:
                raise subprocess.CalledProcessError(2, command, stderr=b"injected Git error")
            return original(command, **kw)
        with mock.patch.object(module.subprocess, "run", side_effect=failing):
            with self.assertRaises(subprocess.CalledProcessError):
                module.materialize(root, 32)

    def test_generated_pollution_overlay_and_repeat(self):
        root = self.home / "archive"
        fixture(root)
        ignored = (
            "common/a.o", "common/log.run", "common/core.12", "common/cache.pyc",
            "common/generated.log", "common/__pycache__/cached.py",
            "common/.generated/old.H", "common/processor0/state",
            "applications/CHMT/.build/native/generated.H",
            "applications/gasUGKP/private_backend/build/linux/old.H",
            "applications/CHT/Make/linux64Gcc/old.H", "tool/build_logs/old.H",
            "validation/composition_transport/evidence/result.json",
        )
        for name in ignored:
            put(root, name, "not source")
        for name in ignored:
            put(root / "precision/fp32", name, "not overlay source")
        a = manifest(root, 32)
        for name in ignored:
            self.assertNotIn(name, a["files"])
        for name in ("Types.H", "Configuration.H", "HostState.H"):
            rel = "applications/CHMT/core/" + name
            self.assertIn(rel, a["files"])
            self.assertEqual(a["files"][rel], hashlib.sha256(
                (root / "precision/fp32" / rel).read_bytes()).hexdigest())
        self.assertIn("validation/build/check_local.py", a["files"])
        self.assertIn("validation/build/check_fp32.py", a["files"])
        self.assertIn("validation/build/evidence/historical.log.txt", a["files"])
        destination = root / ".build/precision/fp32/source"
        self.assertEqual((destination / "common/Choice.H").read_text(),
                         (root / "precision/fp32/common/Choice.H").read_text())
        b = manifest(root, 32)
        self.assertEqual(a["files"], b["files"])
        (root / "common/remove.H").unlink()
        (root / "precision/fp32/common/Choice.H").unlink()
        c = manifest(root, 32)
        self.assertNotIn("common/remove.H", c["files"])
        self.assertFalse((destination / "common/remove.H").exists())
        self.assertEqual((destination / "common/Choice.H").read_text(),
                         (root / "common/Choice.H").read_text())

    def test_directory_links_are_entries_not_traversed(self):
        root = self.home / "archive"
        fixture(root)
        outside = self.home / "outside"
        outside.mkdir()
        put(outside, "must-not-copy.H")
        (root / "common/alias").symlink_to(outside, target_is_directory=True)
        (root / "gpu").symlink_to(outside, target_is_directory=True)
        (root / "precision/fp32/common/alias32").symlink_to(outside, target_is_directory=True)
        (root / "common/BaseLink.H").symlink_to("Base.H")
        data = manifest(root, 32)
        for name in ("common/alias", "common/alias32", "common/BaseLink.H", "gpu"):
            self.assertIn(name, data["files"])
            self.assertTrue((root / ".build/precision/fp32/source" / name).is_symlink())
        self.assertFalse(any("must-not-copy" in name for name in data["files"]))
        self.assertEqual((outside / "must-not-copy.H").read_text(), "source\n")

    def test_tracked_ignored_sources_preserved_untracked_overlay_filtered(self):
        root = self.home / "checkout"
        fixture(root)
        repository(root)
        for name in ("common/kept.log", "precision/fp32/common/kept32.log"):
            put(root, name, "intentionally tracked")
            git(root, "add", "-f", name)
        put(root, "precision/fp32/common/untracked.log", "output")
        put(root, "precision/fp32/applications/CHMT/.build/cache.H", "output")
        data = manifest(root, 32)
        self.assertIn("common/kept.log", data["files"])
        self.assertIn("common/kept32.log", data["files"])
        self.assertNotIn("common/untracked.log", data["files"])
        self.assertNotIn("applications/CHMT/.build/cache.H", data["files"])

if __name__ == "__main__":
    unittest.main()
