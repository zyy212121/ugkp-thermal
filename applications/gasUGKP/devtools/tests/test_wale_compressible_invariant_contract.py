#!/usr/bin/env python3
                                                                   

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"
LES_ALGEBRA = ROOT.parents[1] / "common" / "gasNumerics" / "GpuLesAlgebra.cuh"


class WaleCompressibleInvariantContract(unittest.TestCase):
    def test_wale_denominator_uses_full_symmetric_gradient(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        algebra = LES_ALGEBRA.read_text(encoding="utf-8")
        self.assertIn("double symmetricGradientSquared = 0.0;", source)
        self.assertIn(
            "symmetricGradientSquared +=\n"
            "                symmetricGradient*symmetricGradient;",
            source,
        )
        self.assertIn(
            "pow(fmax(symmetricGradientSquared, 0.0), 2.5)",
            algebra,
        )

    def test_smagorinsky_keeps_deviatoric_strain_invariant(self) -> None:
        algebra = LES_ALGEBRA.read_text(encoding="utf-8")
        self.assertIn(
            "*sqrt(fmax(2.0*deviatoricStrainSquared, 0.0));",
            algebra,
        )


if __name__ == "__main__":
    unittest.main()
