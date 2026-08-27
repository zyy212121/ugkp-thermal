#!/usr/bin/env python3
                                                                             

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "diluteUgkwpFoam.C"
SCHEDULING = ROOT.parents[1] / "common" / "GpuSchedulingConfiguration.H"


class CourantUpdateContract(unittest.TestCase):
    def test_only_two_time_loop_update_calls_exist(self) -> None:
        source = SOLVER.read_text(encoding="utf-8")
        self.assertEqual(source.count("resident.computeGasCourant"), 2)
        self.assertEqual(source.count("const bool courantRefreshDue"), 2)

    def test_update_condition_is_interval_only(self) -> None:
        source = SOLVER.read_text(encoding="utf-8")
        conditions = re.findall(
            r"const bool courantRefreshDue\s*=([\s\S]*?);\n\s*if \(courantRefreshDue\)",
            source,
        )
        self.assertEqual(len(conditions), 2)
        for condition in conditions:
            self.assertEqual(
                " ".join(condition.split()),
                "adjustTimeStep && stepsSinceCourant >= gpuResidentCourantUpdateInterval",
            )

    def test_default_interval_is_1000(self) -> None:
        source = SCHEDULING.read_text(encoding="utf-8")
        self.assertIn("label courantUpdateInterval = 1000;", source)
        self.assertRegex(
            source,
            r'lookupOrDefault<label>\s*\(\s*"gpuResidentCourantUpdateInterval",\s*1000\s*\)',
        )

    def test_counter_represents_exact_completed_step_interval(self) -> None:
        source = SOLVER.read_text(encoding="utf-8")
        self.assertEqual(source.count("stepsSinceCourant = 1;"), 2)
        self.assertNotIn("stepsSinceCourant = 0;", source)


if __name__ == "__main__":
    unittest.main()
