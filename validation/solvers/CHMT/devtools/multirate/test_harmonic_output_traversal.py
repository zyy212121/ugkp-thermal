#!/usr/bin/env python3
"""Source-level traversal guard; numerical equivalence is checked by harmonic.

This deliberately counts output loops rather than timings: the bounded change
removes redundant traversal, without promising a compiler-dependent speedup.
"""
from pathlib import Path
import re
import sys

VALIDATION_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "validation")
sys.path.insert(0, str(VALIDATION_ROOT))
from tools.paths import SOURCE_ROOT
import unittest


class HarmonicOutputTraversal(unittest.TestCase):
    def test_output_has_one_point_traversal(self):
        header = SOURCE_ROOT / "applications/CHMT/mesh/SweepWorkspace.H"
        source = header.read_text()
        output = source.split("output.resize(mesh_.points.size());", 1)[1]
        output = output.split("return true;", 1)[0]
        traversals = len(re.findall(r"\bfor\s*\(", output))
        self.assertEqual(traversals, 1,
                         "harmonic output must use one point traversal")


if __name__ == "__main__":
    unittest.main()
