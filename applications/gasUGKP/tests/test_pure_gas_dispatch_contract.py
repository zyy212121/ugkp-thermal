#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "diluteUgkwpFoam.C"


class PureGasDispatchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOLVER.read_text(encoding="utf-8")

    def test_runtime_dictionary_switch_is_read(self):
        self.assertIn(
            "const bool gpuResidentPureGasOnly = gpuScheduling.pureGasOnly;",
            self.source,
        )

    def test_runtime_switch_dispatches_to_pure_gas_solver(self):
        self.assertRegex(
            self.source,
            re.compile(
                r"if\s*\(\s*gpuResidentPureGasOnly\s*\)\s*\{\s*"
                r"GpuGasResidentSolver\s+resident\s*;",
                re.DOTALL,
            ),
        )
        self.assertNotRegex(
            self.source,
            r"if\s*\(\s*false\s*/\*\s*gpuResidentPureGasOnly\s*\*/\s*\)",
        )


if __name__ == "__main__":
    unittest.main()
