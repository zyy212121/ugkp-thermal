                                                                       

from __future__ import annotations

import unittest
from pathlib import Path

try:
    from .source_contract_utils import code_only, function_block
except ImportError:
    from source_contract_utils import code_only, function_block


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"
PROTOCOL = ROOT / "gpu" / "GpuBackendProtocol.H"
CREATE_FIELDS = ROOT / "createFields.H"
GAS_CONFIGURATION = ROOT / "readGpuGasConfiguration.H"
INTERPOLATION_HEADER = (
    ROOT / "private_backend" / "OpenFoamLimitedLinear.cuh"
)


class OpenFoamEnergyInterpolationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.face_flux = code_only(
            function_block(cls.source, "computeRiemannGasFaceFluxDevice")
        )

    def test_openfoam_limited_linear_helper_exists_and_is_used(self) -> None:
        self.assertTrue(INTERPOLATION_HEADER.is_file())
        self.assertIn('#include "OpenFoamLimitedLinear.cuh"', self.source)
        header = INTERPOLATION_HEADER.read_text(encoding="utf-8")
        self.assertIn("limitedLinearFaceValue", header)
        self.assertIn(
            "ugkpinterpolation::limitedLinearFaceValue",
            self.face_flux,
        )

    def test_energy_only_mode_is_traceable_across_frontend_protocol(self) -> None:
        protocol = PROTOCOL.read_text(encoding="utf-8")
        create_fields = (
            GAS_CONFIGURATION.read_text(encoding="utf-8")
            + "\n"
            + CREATE_FIELDS.read_text(encoding="utf-8")
        )
        self.assertIn("OpenFoamEnergyLimitedLinear", protocol)
        self.assertIn("energyLimitedLinear", create_fields)
        self.assertIn("gasReconstruction = 2", create_fields)

    def test_energy_mode_keeps_inviscid_face_states_first_order(self) -> None:
        reconstruction = code_only(
            function_block(self.source, "reconstructGasCellToFace")
        )
        self.assertIn("s.gasReconstruction != 1", reconstruction)
        self.assertIn("s.gasReconstruction == 2", self.face_flux)

    def test_energy_mode_is_available_for_every_riemann_flux(self) -> None:
        configuration = GAS_CONFIGURATION.read_text(encoding="utf-8")
        self.assertNotIn(
            "gasReconstruction == 2\n && gasFluxScheme != 8",
            configuration,
        )
        energy_guard = self.face_flux[
            self.face_flux.index("s.gasReconstruction == 2") :
            self.face_flux.index(
                "const double ownerVelocitySquared",
                self.face_flux.index("s.gasReconstruction == 2"),
            )
        ]
        self.assertNotIn("Scheme::SLAU2", energy_guard)
        self.assertNotIn("Scheme::SLAU2_2", energy_guard)

    def test_internal_energy_terms_are_limited_field_by_field(self) -> None:
                                                               
                                                                            
                                                                            
        self.assertGreaterEqual(
            self.face_flux.count(
                "ugkpinterpolation::limitedLinearFaceValue"
            ),
            2,
        )
        self.assertIn("faceThermalEnthalpy", self.face_flux)
        self.assertIn("faceKineticEnergy", self.face_flux)
        self.assertIn(
            "faceThermalEnthalpy + faceKineticEnergy",
            self.face_flux,
        )
        self.assertIn(
            "ugkpinterpolation::limitedLinearRiemannEnergyFlux",
            self.face_flux,
        )
        self.assertNotIn("ownerEnthalpy", self.face_flux)

    def test_particle_functions_do_not_use_energy_interpolation_helper(self) -> None:
        for name in (
            "applyEulerianGasSolidCouplingKernel",
            "trackParticlesLocalFaceWalkKernel",
            "samplePoissonPoolParticlesKernel",
        ):
            with self.subTest(function=name):
                body = function_block(self.source, name)
                self.assertNotIn("ugkpinterpolation::", body)


if __name__ == "__main__":
    unittest.main()
