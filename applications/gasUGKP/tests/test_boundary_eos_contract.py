#!/usr/bin/env python3

import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "gpu" / "GpuResidentStrict.H"


def relative_eos_error(rho: float, pressure: float, temperature: float, gas_r: float) -> float:
    expected = pressure / (rho * gas_r)
    return abs(temperature - expected) / max(abs(temperature), abs(expected), 1.0)


class BoundaryEosAlgebra(unittest.TestCase):
    def test_consistent_triple_is_accepted(self) -> None:
        rho = 1.2
        gas_r = 287.0
        temperature = 900.0
        pressure = rho * gas_r * temperature
        self.assertLessEqual(relative_eos_error(rho, pressure, temperature, gas_r), 1.0e-8)

    def test_inconsistent_triple_is_rejected(self) -> None:
        self.assertGreater(relative_eos_error(1.2, 101325.0, 900.0, 287.0), 1.0e-8)


class BoundaryEosSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = HEADER.read_text(encoding="utf-8")

    def test_upload_receives_the_solver_gas_constant(self) -> None:
        self.assertGreaterEqual(
            self.source.count(
                "RgasValue,\n            strictOpenFoamGasBoundaries"
            ),
            2,
        )

    def test_inconsistent_three_fixed_values_fail_before_upload(self) -> None:
        for token in (
            "rhoFixes != 0",
            "pFixes != 0",
            "TFixes != 0",
            "relativeEosError > scalar(1e-8)",
            "violates p=rho*R*T",
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
