#!/usr/bin/env python3
                                                                   

from __future__ import annotations
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


import unittest
from pathlib import Path


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')
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
            "pow(fmax(symmetricGradientSquared, GPU_R(0.0)), GPU_R(2.5))",
            algebra,
        )

    def test_smagorinsky_keeps_deviatoric_strain_invariant(self) -> None:
        algebra = LES_ALGEBRA.read_text(encoding="utf-8")
        self.assertIn(
            "*sqrt(fmax(GPU_R(2.0)*deviatoricStrainSquared, GPU_R(0.0)));",
            algebra,
        )


if __name__ == "__main__":
    unittest.main()
