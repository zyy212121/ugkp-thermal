#!/usr/bin/env python3
"""Source-level traversal guard; numerical equivalence is checked by harmonic.

This deliberately counts output loops rather than timings: the bounded change
removes redundant traversal, without promising a compiler-dependent speedup.
"""
from pathlib import Path
import re
import unittest


class HarmonicOutputTraversal(unittest.TestCase):
    def test_output_has_one_point_traversal(self):
        header = Path(__file__).resolve().parents[2] / "mesh/SweepWorkspace.H"
        source = header.read_text()
        output = source.split("output.resize(mesh_.points.size());", 1)[1]
        output = output.split("return true;", 1)[0]
        traversals = len(re.findall(r"\bfor\s*\(", output))
        self.assertEqual(traversals, 1,
                         "harmonic output must use one point traversal")


if __name__ == "__main__":
    unittest.main()
