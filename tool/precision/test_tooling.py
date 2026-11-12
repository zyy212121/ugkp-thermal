#!/usr/bin/env python3
"""Non-numerical tests of source isolation and process startup selection."""
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("materialize_sources", ROOT / "tool/precision/materialize_sources.py")
materializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materializer)

class PrecisionToolingTests(unittest.TestCase):
    def test_materialized_sources_are_independent(self):
        tree64 = materializer.materialize(ROOT, 64)
        tree32 = materializer.materialize(ROOT, 32)
        self.assertIn("compiledBits = 64", (tree64 / "common/PrecisionBuild.H").read_text())
        self.assertIn("compiledBits = 32", (tree32 / "common/PrecisionBuild.H").read_text())
        original = ROOT / "common/GpuPrecisionTypes.H"
        self.assertEqual((tree64 / "common/GpuPrecisionTypes.H").read_bytes(), original.read_bytes())
        overlay = ROOT / "precision/fp32"
        for source in overlay.rglob("*"):
            if source.is_file() and not source.is_symlink():
                self.assertEqual((tree32 / source.relative_to(overlay)).read_bytes(), source.read_bytes())
        for tree in (tree64, tree32):
            for link in tree.rglob("*"):
                if link.is_symlink():
                    self.assertTrue(link.resolve().is_relative_to(tree), str(link))
        self.assertNotEqual(os.stat(tree64 / "common/GpuPrecisionTypes.H").st_ino,
                            os.stat(tree32 / "common/GpuPrecisionTypes.H").st_ino)

    def test_launcher_selects_once_and_preserves_arguments(self):
        launcher = ROOT / ".build/precision/launcher/ugkpPrecisionLauncher"
        self.assertTrue(launcher.is_file(), "Build the host-only launcher first.")
        with tempfile.TemporaryDirectory(prefix="launcher-test-", dir=ROOT / ".build/precision") as temp:
            root = Path(temp)
            bindir = root / "bin"
            case = root / "case"
            (case / "constant").mkdir(parents=True)
            bindir.mkdir()
            for bits in (32, 64):
                target = bindir / ".precision" / str(bits)
                target.mkdir(parents=True)
                for solver in ("gasUGKP", "gasUGKPSteady", "CHT", "CHMT"):
                    fake = target / solver
                    fake.write_text("#!/bin/sh\nprintf 'FAKE_MODE=" + str(bits) + "\\n'\nprintf '<%s>\\n' \"$@\"\n")
                    fake.chmod(0o755)
                    shutil.copy2(launcher, bindir / solver)
            config = case / "constant/scheme"
            for solver in ("gasUGKP", "gasUGKPSteady", "CHT", "CHMT"):
                for value, expected in ((None, 32), ("precision 64;", 64), ("precision 32;", 32),
                                        ("precision 16;", 32), ("precision broken;", 32),
                                        ("wallLayer { maxIterations 77; }", 32),
                                        ("precision 64; wallLayer { maxIterations 77; }", 64),
                                        ("precision 32; solid { nonlinear { nonlinearMaxIterations 19; } }", 32)):
                    if value is None:
                        config.unlink(missing_ok=True)
                    else:
                        config.write_text(value + "\n")
                    result = subprocess.run([str(bindir / solver), "-case", str(case),
                                             "-unchangedOption", "value with spaces"],
                                            text=True, capture_output=True)
                    print("STUB", solver, repr(value), "expected", expected, "exit", result.returncode, result.stdout.strip(), result.stderr.strip())
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("FAKE_MODE=" + str(expected), result.stdout)
                    self.assertIn("<value with spaces>", result.stdout)
                    self.assertIn("<" + str(case) + ">", result.stdout)
            config.unlink(missing_ok=True)
            (case / "constant/precisionProperties").write_text("precision 64;\n")
            result = subprocess.run([str(bindir / "CHMT"), "-case", str(case)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FAKE_MODE=32", result.stdout)
            print("STUB obsolete-file-ignored", result.stdout.strip())
            (case / "constant/precisionChoice").write_text("precision 64;\n")
            config.write_text('#include "precisionChoice"\n')
            result = subprocess.run([str(bindir / "gasUGKP"), "-case", str(case)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FAKE_MODE=64", result.stdout)

if __name__ == "__main__":
    unittest.main()
