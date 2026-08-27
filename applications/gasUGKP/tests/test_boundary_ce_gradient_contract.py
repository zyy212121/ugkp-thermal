                                                                        

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

BOUNDARY_FIELDS = (
    "Kind",
    "RhoFix",
    "UFix",
    "PFix",
    "TFix",
    "PWave",
    "PWaveGamma",
    "PWaveFieldInf",
    "PWaveLInf",
    "Rho",
    "Ux",
    "Uy",
    "Uz",
    "P",
    "T",
)


def close_wave_pressure_state(
    owner_pressure: float,
    owner_temperature: float,
    wave_pressure: float,
    gas_constant: float,
) -> tuple[float, float, float]:
                                                                         

    del owner_pressure
    pressure = wave_pressure
    temperature = owner_temperature
    density = pressure / (gas_constant * temperature)
    return density, pressure, temperature


class BoundaryMirrorSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.active_code = code_only(cls.source)
        cls.allocate = function_block(cls.source, "allocateFields")
        cls.release = function_block(cls.source, "releaseState")
        cls.upload = function_block(
            cls.source, "ugkwpGpuResidentStrictUploadGasBoundaryFields"
        )

    def test_legacy_and_riemann_boundary_arrays_have_independent_storage(
        self,
    ) -> None:
        for suffix in BOUNDARY_FIELDS:
            with self.subTest(field=suffix):
                value_type = "int" if suffix in {
                    "Kind", "RhoFix", "UFix", "PFix", "TFix", "PWave"
                } else "double"
                for prefix in ("gasBoundary", "riemannBoundary"):
                    field = f"{prefix}{suffix}"
                    self.assertRegex(
                        self.active_code,
                        rf"\b{value_type}\s*\*\s*{field}\s*=\s*nullptr\s*;",
                    )
                    self.assertRegex(
                        self.allocate,
                        rf"allocate\s*\(\s*s->{field}\s*,\s*nf\s*,",
                    )
                    self.assertRegex(
                        self.release,
                        rf"release\s*\(\s*s->{field}\s*\)\s*;",
                    )

    def test_upload_synchronizes_both_boundary_mirrors_field_by_field(
        self,
    ) -> None:
        for suffix in BOUNDARY_FIELDS:
            argument = f"gasBoundary{suffix}"
            with self.subTest(field=suffix):
                for destination in (
                    f"gasBoundary{suffix}",
                    f"riemannBoundary{suffix}",
                ):
                    self.assertRegex(
                        self.upload,
                        rf"copyToDevice\s*\(\s*s->{destination}\s*,\s*"
                        rf"{argument}\s*,\s*nf\s*,",
                    )

    def test_runtime_boundary_updates_keep_pressure_mirrors_synchronized(
        self,
    ) -> None:
        wave_update = function_block(
            self.source, "updateWaveTransmissivePressureBoundaryKernel"
        )
        scheduled_update = function_block(
            self.source, "updateLegacyGasBoundaryMirrorKernel"
        )
        for body in (wave_update, scheduled_update):
            self.assertIn("s.gasBoundaryP[", body)
            self.assertIn("s.riemannBoundaryP[", body)


class RiemannBoundaryGradientSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.boundary_state = function_block(
            cls.source, "riemannBoundaryState"
        )
        cls.face_gradient = function_block(
            cls.source, "riemannFacePrimitiveForGradient"
        )
        cls.gradient_kernel = function_block(
            cls.source, "computeGasPrimitiveGradientsKernel"
        )
        cls.limiter_kernel = function_block(
            cls.source, "computeGasGradientLimiterKernel"
        )
        cls.face_flux = function_block(
            cls.source, "computeRiemannGasFaceFluxDevice"
        )
        cls.courant_kernel = function_block(
            cls.source, "computeGasCourantFieldKernel"
        )
        cls.cell_courant_kernel = function_block(
            cls.source, "computeGasConvectiveCourantByCellKernel"
        )

    def test_wave_pressure_is_a_fixed_pressure_in_shared_boundary_state(
        self,
    ) -> None:
        self.assertRegex(
            self.boundary_state,
            r"const\s+bool\s+pFixed\s*=\s*"
            r"s\.riemannBoundaryPFix\[f\]\s*!=\s*0\s*\|\|\s*"
            r"s\.riemannBoundaryPWave\[f\]\s*!=\s*0\s*;",
        )
        self.assertIn(
            "return riemannBoundaryState(s, f, centre);",
            self.face_gradient,
        )
        self.assertIn(
            "riemannFacePrimitiveForGradient(s, c, f)",
            self.gradient_kernel,
        )
        self.assertIn(
            "riemannFacePrimitiveForGradient(s, c, f)",
            self.limiter_kernel,
        )

    def test_flux_and_gradient_boundary_path_do_not_read_legacy_arrays(
        self,
    ) -> None:
        active_gas_functions = (
            "useRiemannBoundaryVelocity",
            "riemannBoundaryState",
            "riemannExteriorStateForFace",
            "riemannFacePrimitiveForGradient",
            "computeGasPrimitiveGradientsKernel",
            "computeGasGradientLimiterKernel",
            "computeRiemannGasFaceFluxDevice",
            "computeGasCourantFieldKernel",
        )
        for name in active_gas_functions:
            with self.subTest(function=name):
                body = code_only(function_block(self.source, name))
                self.assertNotRegex(body, r"\bs\.gasBoundary[A-Z]")

    def test_convective_courant_matches_rho_central_cell_sum(self) -> None:
        self.assertIn("spectralRadius*area", self.courant_kernel)
        self.assertNotIn("deltaCoeffs", self.courant_kernel)
        self.assertIn(
            "sumAmaxSf += finiteOr(s.gasPhiRho[f], OfGreat);",
            self.cell_courant_kernel,
        )
        self.assertRegex(
            self.cell_courant_kernel,
            r"0\.5\s*\*\s*dt\s*\*\s*sumAmaxSf\s*/\s*"
            r"clampMin\(s\.V\[c\],\s*OfSmall\)",
        )

    def test_boundary_state_has_one_ideal_gas_closure_for_flux_and_gradient(
        self,
    ) -> None:
        for branch in (
            "if (pFixed && TFixed)",
            "else if (rhoFixed && TFixed)",
            "else if (rhoFixed && pFixed)",
            "else if (pFixed)",
            "else if (rhoFixed)",
            "else if (TFixed)",
        ):
            self.assertIn(branch, self.boundary_state)
        self.assertIn(
            "return makeGasPrimDevice",
            self.boundary_state,
        )
        self.assertIn(
            "riemannExteriorStateForFace(s, f, left)",
            self.face_flux,
        )

    def test_wedge_symmetry_is_not_a_propagating_courant_direction(
        self,
    ) -> None:
                                                                             

        active = code_only(self.courant_kernel)
        self.assertRegex(
            active,
            r"f\s*>=\s*s\.nInternalFaces\s*&&\s*\(\s*"
            r"s\.riemannBoundaryKind\[f\]\s*==\s*1\s*\|\|\s*"
            r"s\.riemannBoundaryKind\[f\]\s*==\s*3\s*\|\|\s*"
            r"s\.riemannBoundaryKind\[f\]\s*==\s*4\s*\)",
        )
        self.assertNotRegex(
            active,
            r"s\.riemannBoundaryKind\[f\]\s*==\s*2\s*\|\|",
        )


class RiemannBoundaryAlgebraRegression(unittest.TestCase):
    def test_pwave_state_used_by_gradient_is_thermodynamically_closed(
        self,
    ) -> None:
        rho, pressure, temperature = close_wave_pressure_state(
            owner_pressure=100_000.0,
            owner_temperature=300.0,
            wave_pressure=92_000.0,
            gas_constant=287.0,
        )
        self.assertEqual(pressure, 92_000.0)
        self.assertEqual(temperature, 300.0)
        self.assertAlmostEqual(pressure, rho * 287.0 * temperature)
        self.assertNotAlmostEqual(rho, 100_000.0 / (287.0 * 300.0))


if __name__ == "__main__":
    unittest.main()
