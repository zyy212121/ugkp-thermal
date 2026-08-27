                                                                 

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    from .source_contract_utils import branch_block, code_only, function_block
except ImportError:
    from source_contract_utils import branch_block, code_only, function_block


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "private_backend" / "GpuResidentStrict.cu"


def direct_wall_euler_flux(
    pressure: float,
    normal: tuple[float, float, float],
    wall_velocity: tuple[float, float, float],
    wall_temperature: float,
) -> tuple[float, tuple[float, float, float], float]:
                                                                               

    del wall_temperature
    momentum = tuple(pressure * component for component in normal)
    pressure_work = sum(
        momentum_component * velocity_component
        for momentum_component, velocity_component in zip(momentum, wall_velocity)
    )
    return 0.0, momentum, pressure_work


def zero_transport_correction(
    gas_viscosity: float,
    turbulent_viscosity: float,
    gas_cp: float,
    gas_prandtl: float,
    turbulent_conductivity: float,
) -> tuple[float, float]:
                                                                               

    mu_effective = gas_viscosity + turbulent_viscosity
    k_effective = (
        gas_viscosity * gas_cp / gas_prandtl + turbulent_conductivity
    )
    return mu_effective, k_effective


class RiemannWallSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.face_flux = function_block(
            cls.source, "computeRiemannGasFaceFluxDevice"
        )
        cls.wall_branch = branch_block(
            cls.face_flux,
            r"(?:else\s+)?if\s*\(\s*boundaryKind\s*==\s*2\s*\)",
        )
        cls.symmetry_branch = branch_block(
            cls.face_flux,
            r"if\s*\(\s*boundaryKind\s*==\s*1\s*\)",
        )

    def test_wall_and_symmetry_are_direct_euler_boundary_fluxes(self) -> None:
        self.assertRegex(self.face_flux, r"double\s+massFlux\s*=\s*0\.0\s*;")
        for branch in (self.wall_branch, self.symmetry_branch):
            self.assertNotRegex(branch, r"\bmassFlux\s*=")
            self.assertNotIn("fluxUnitArea", branch)
            for component, normal in (
                ("X", "nx"),
                ("Y", "ny"),
                ("Z", "nz"),
            ):
                self.assertRegex(
                    branch,
                    rf"momentumFlux{component}\s*=\s*left\.p\s*\*\s*{normal}\s*;",
                )

    def test_wall_pressure_does_not_depend_on_wall_thermal_or_tangent_state(
        self,
    ) -> None:
        for component in ("X", "Y", "Z"):
            assignment = re.search(
                rf"momentumFlux{component}\s*=\s*(?P<rhs>[^;]+);",
                self.wall_branch,
            )
            self.assertIsNotNone(assignment)
            rhs = assignment.group("rhs")
            self.assertNotRegex(rhs, r"\bright\b|\bwallU|\bT\b|Boundary")

    def test_no_maxwell_full_velocity_reflection_remains_in_active_gas_path(
        self,
    ) -> None:
        active = "\n".join(
            code_only(function_block(self.source, name))
            for name in (
                "riemannBoundaryState",
                "riemannExteriorStateForFace",
                "riemannFacePrimitiveForGradient",
                "computeRiemannGasFaceFluxDevice",
            )
        )
        self.assertNotRegex(
            active,
            r"(?:right|wall)\.(?:ux|uy|uz)\s*=\s*"
            r"2(?:\.0)?\s*\*[^;]+-\s*left\.(?:ux|uy|uz)",
        )
        self.assertNotIn("Maxwell", active)

    def test_symmetry_has_no_viscous_or_fourier_transport(self) -> None:
        self.assertRegex(
            self.face_flux,
            r"if\s*\(\s*boundaryKind\s*!=\s*1\s*\)",
        )
        self.assertNotIn("muEffective", self.symmetry_branch)
        self.assertNotIn("kEffective", self.symmetry_branch)

    def test_wall_viscous_shear_and_heat_are_a_separate_additive_path(
        self,
    ) -> None:
        self.assertIn("const double muEffective = s.gasMu + muTurbulent;", self.face_flux)
        self.assertIn(
            "molecularGasConductivity(s) + kTurbulent", self.face_flux
        )
        self.assertRegex(
            self.face_flux,
            r"if\s*\(\s*muEffective\s*>\s*0\.0\s*\|\|\s*"
            r"kEffective\s*>\s*0\.0\s*\)",
        )
        for correction in (
            "momentumFluxX -= traction.x;",
            "momentumFluxY -= traction.y;",
            "momentumFluxZ -= traction.z;",
            "energyFlux -= kEffective*normalTemperatureGradient;",
        ):
            self.assertIn(correction, self.face_flux)

    def test_empty_and_processor_faces_are_skipped_before_flux_work(self) -> None:
        self.assertRegex(
            self.face_flux,
            r"if\s*\(\s*boundaryKind\s*==\s*3\s*\|\|\s*"
            r"boundaryKind\s*==\s*4\s*\)\s*\{\s*return\s+false\s*;",
        )
        skip_position = self.face_flux.index("boundaryKind == 3")
        reconstruction_position = self.face_flux.index(
            "reconstructGasCellToFace"
        )
        self.assertLess(skip_position, reconstruction_position)


class RiemannWallAlgebraRegression(unittest.TestCase):
    def test_tangential_speed_and_temperature_do_not_raise_normal_pressure(
        self,
    ) -> None:
        base = direct_wall_euler_flux(
            pressure=100_000.0,
            normal=(1.0, 0.0, 0.0),
            wall_velocity=(0.0, 0.0, 0.0),
            wall_temperature=300.0,
        )
        hot_fast_tangent = direct_wall_euler_flux(
            pressure=100_000.0,
            normal=(1.0, 0.0, 0.0),
            wall_velocity=(0.0, 250.0, -175.0),
            wall_temperature=1200.0,
        )
        self.assertEqual(base, hot_fast_tangent)
        self.assertEqual(hot_fast_tangent[0], 0.0)
        self.assertEqual(hot_fast_tangent[1], (100_000.0, 0.0, 0.0))
        self.assertEqual(hot_fast_tangent[2], 0.0)

    def test_zero_molecular_and_sgs_transport_produces_no_correction(self) -> None:
        self.assertEqual(
            zero_transport_correction(
                gas_viscosity=0.0,
                turbulent_viscosity=0.0,
                gas_cp=1004.5,
                gas_prandtl=0.72,
                turbulent_conductivity=0.0,
            ),
            (0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
