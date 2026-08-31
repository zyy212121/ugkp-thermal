                                                                      

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    from .source_contract_utils import code_only, function_block
except ImportError:
    from source_contract_utils import code_only, function_block


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"
TRANSPORT_HEADER = ROOT / "private_backend" / "OpenFoamViscousFlux.cuh"


class OpenFoamViscousFluxSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.face_flux = code_only(
            function_block(cls.source, "computeRiemannGasFaceFluxDevice")
        )

    def test_openfoam_transport_header_exists_and_is_used(self) -> None:
        self.assertTrue(TRANSPORT_HEADER.is_file())
        self.assertIn('#include "OpenFoamViscousFlux.cuh"', self.source)
        header = TRANSPORT_HEADER.read_text(encoding="utf-8")
        self.assertIn("makeInternalSnGradGeometry", header)
        self.assertIn("correctedSnGrad", header)
        self.assertIn("openFoamNewtonianTraction", header)

    def test_internal_heat_flux_uses_compact_corrected_sn_grad(self) -> None:
        self.assertIn(
            "ugkptransport::correctedSnGrad",
            self.face_flux,
        )
        self.assertNotRegex(
            self.face_flux,
            re.compile(
                r"gradT[XYZ]\s*=\s*0\.5\s*\*\s*\("
                r"\s*gradT[XYZ]\s*\+\s*s\.gradT[XYZ]\[nei\]",
                flags=re.DOTALL,
            ),
        )

    def test_velocity_stress_uses_openfoam_split_operator(self) -> None:
        self.assertIn(
            "ugkptransport::openFoamNewtonianTraction",
            self.face_flux,
        )
        self.assertGreaterEqual(
            self.face_flux.count("ugkptransport::correctedSnGrad"),
            4,
        )
        self.assertNotIn(
            "ugkptransport::normalConsistentGradient",
            self.face_flux,
        )
        self.assertIn("s.faceWeight[f]", self.face_flux)

    def test_particle_functions_do_not_use_transport_helper(self) -> None:
        for name in (
            "applyGasVolumeFractionSourceKernel",
            "applyEulerianGasSolidDragKernelStatic",
            "applyEulerianParticleMaterialHeatKernel",
            "trackParticlesLocalFaceWalkKernel",
            "samplePoissonPoolParticlesKernel",
        ):
            with self.subTest(function=name):
                body = function_block(self.source, name)
                self.assertNotIn("ugkptransport::", body)


if __name__ == "__main__":
    unittest.main()
