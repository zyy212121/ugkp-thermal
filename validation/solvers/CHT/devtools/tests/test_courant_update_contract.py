#!/usr/bin/env python3
                                                                             

from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


import re
import unittest
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/CHT')
SOLVER = ROOT / "diluteUgkwpFoam.C"
CASE_PROPERTIES = (
    (CASES_ROOT / 'example/production/MSS7_laminar/constant/schedulingProperties')
)
SCHEDULING_HEADER = ROOT.parents[1] / "common/GpuSchedulingConfiguration.H"


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
            compact = " ".join(condition.split())
            self.assertEqual(
                compact,
                "adjustTimeStep && stepsSinceCourant >= gpuResidentCourantUpdateInterval",
            )

    def test_default_and_mss7_interval_are_1000(self) -> None:
        source = SCHEDULING_HEADER.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r'"gpuResidentCourantUpdateInterval",\s*1000',
        )
        properties = CASE_PROPERTIES.read_text(encoding="utf-8")
        self.assertRegex(
            properties,
            r"gpuResidentCourantUpdateInterval\s+1000;",
        )

    def test_counter_represents_exact_completed_step_interval(self) -> None:
        source = SOLVER.read_text(encoding="utf-8")
        self.assertEqual(source.count("stepsSinceCourant = 1;"), 2)
        self.assertNotIn("stepsSinceCourant = 0;", source)


if __name__ == "__main__":
    unittest.main()
