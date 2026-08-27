                                                                      

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "gpu" / "GpuGasBoundarySpec.H"
WRAPPER = ROOT / "gpu" / "GpuResidentStrict.H"
CONFIG = ROOT / "readGpuGasConfiguration.H"
SOLVER = ROOT / "diluteUgkwpFoam.C"


class OpenFoamBoundaryConfigurationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = SPEC.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.config = CONFIG.read_text(encoding="utf-8")
        cls.solver = SOLVER.read_text(encoding="utf-8")

    def test_geometry_and_field_operators_are_separate(self) -> None:
        self.assertIn("enum class PatchGeometry", self.spec)
        self.assertIn("enum class ScalarOperator", self.spec)
        self.assertIn("enum class VelocityOperator", self.spec)
        self.assertIn('#include "GpuGasBoundarySpec.H"', self.wrapper)

    def test_native_cases_enable_strict_openfoam_boundary_semantics(self) -> None:
        self.assertNotIn("gksProps", self.config)
        self.assertNotIn("particleProperties/gks", self.config)
        self.assertNotIn("Using deprecated gas", self.config)
        self.assertRegex(
            self.solver,
            r"Rgas\.value\(\),\s*true,\s*muG\.value\(\)",
        )

    def test_frontend_rejects_unsupported_patch_field_operators(self) -> None:
        self.assertIn("ugkpboundary::scalarOperator", self.wrapper)
        self.assertIn("ugkpboundary::velocityOperator", self.wrapper)
        self.assertIn("Unsupported OpenFOAM gas boundary field", self.wrapper)

    def test_native_wall_velocity_must_be_dirichlet(self) -> None:
        self.assertIn("wall patch requires noSlip or fixedValue U", self.wrapper)


if __name__ == "__main__":
    unittest.main()
