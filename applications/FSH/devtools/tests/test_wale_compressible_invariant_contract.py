#!/usr/bin/env python3
                                                                      

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"
LES_ALGEBRA = ROOT.parents[1] / "common" / "gasNumerics" / "GpuLesAlgebra.cuh"


class WaleCompressibleInvariantContract(unittest.TestCase):
    def test_full_symmetric_gradient_norm_is_separate_from_smagorinsky_norm(self) -> None:
        source = CUDA_SOURCE.read_text(encoding="utf-8")
        algebra = LES_ALGEBRA.read_text(encoding="utf-8")
        self.assertIn("double symmetricGradientSquared = 0.0;", source)
        self.assertRegex(
            source,
            r"const double symmetricGradient\s*=\s*0\.5\*\(g\[i\]\[j\]\s*\+\s*g\[j\]\[i\]\);"
            r"\s*symmetricGradientSquared\s*\+=\s*symmetricGradient\*symmetricGradient;",
        )
        self.assertRegex(
            algebra,
            r"const double denominator\s*=\s*pow\(fmax\(symmetricGradientSquared, 0\.0\), 2\.5\)"
            r"\s*\+\s*pow\(fmax\(tracelessSquaredGradientSquared, 0\.0\), 1\.25\);",
        )
        self.assertIn("sqrt(fmax(2.0*deviatoricStrainSquared, 0.0))", algebra)


if __name__ == "__main__":
    unittest.main()
