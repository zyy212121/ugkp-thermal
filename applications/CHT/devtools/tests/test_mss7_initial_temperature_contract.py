#!/usr/bin/env python3
                                                                              

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INITIALISER = ROOT / "MSS7_pureGasCHT" / "tools" / "initialise_graphite_temperature.py"


def load_initialiser():
    spec = importlib.util.spec_from_file_location("mss7_initialiser", INITIALISER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {INITIALISER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skip(
    "historical MSS7 preprocessing utility is intentionally excluded from "
    "the minimal runnable validation case"
)
class Mss7InitialTemperatureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_initialiser()

    def test_digitized_depth_knots_are_preserved(self) -> None:
        for depth, temperature in zip(
            self.module.DEPTHS_M, self.module.DEPTH_TEMPERATURES_K
        ):
            self.assertAlmostEqual(
                self.module.measured_depth_temperature(depth), temperature, places=9
            )

    def test_depth_profile_is_monotone_and_continuously_sloped(self) -> None:
        samples = [
            self.module.measured_depth_temperature(i * 0.0001)
            for i in range(401)
        ]
        self.assertTrue(
            all(a + 1.0e-10 >= b for a, b in zip(samples, samples[1:]))
        )
        step = 1.0e-7
        for knot in self.module.DEPTHS_M[1:-1]:
            left = (
                self.module.measured_depth_temperature(knot)
                - self.module.measured_depth_temperature(knot - step)
            ) / step
            right = (
                self.module.measured_depth_temperature(knot + step)
                - self.module.measured_depth_temperature(knot)
            ) / step
            scale = max(1.0, abs(left), abs(right))
            self.assertLess(abs(left - right), max(0.1, 2.0e-4 * scale))

    def test_axial_weight_has_throat_plateau_and_c2_taper(self) -> None:
        throat = self.module.X_THROAT
        plateau = self.module.AXIAL_PLATEAU_HALF_WIDTH_M
        taper = self.module.AXIAL_TAPER_WIDTH_M
        self.assertEqual(self.module.axial_weight(throat), 1.0)
        self.assertEqual(self.module.axial_weight(throat + 0.99 * plateau), 1.0)
        self.assertEqual(
            self.module.axial_weight(throat + plateau + taper), 0.0
        )
        self.assertEqual(
            self.module.axial_weight(throat + plateau + taper + 0.01), 0.0
        )
        midpoint = self.module.axial_weight(throat + plateau + 0.5 * taper)
        self.assertTrue(math.isclose(midpoint, 0.5, rel_tol=0.0, abs_tol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
