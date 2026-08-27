                                                                           

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


class ObsoleteGksPathExclusionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.active_code = code_only(cls.source)
        cls.face_flux = function_block(
            cls.source, "computeRiemannGasFaceFluxDevice"
        )

    def test_active_cuda_has_no_bgk_relaxation_or_tau_wave_selector(self) -> None:
        for obsolete in (
            "tauWave",
            "tauJump",
            "shockRelaxation",
            "ugkpgks",
            "FullSecondOrderGks",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, self.active_code)

    def test_active_flux_is_the_runtime_selected_riemann_flux(self) -> None:
        self.assertIn('#include "RiemannGasFlux.cuh"', self.source)
        self.assertIn(
            "ugkpriemann::schemeFromCreateCode(s.gasFluxScheme, scheme)",
            self.face_flux,
        )
        self.assertIn("ugkpriemann::fluxUnitArea", self.face_flux)

    def test_gas_flux_dissipation_does_not_read_particle_occupancy(self) -> None:
        active_face_flux = code_only(self.face_flux)
        for particle_coupling_token in (
            "epsS",
            "epsGPrev",
            "ownerParticleCoupled",
            "neighbourParticleCoupled",
            "particleCapacity",
        ):
            with self.subTest(token=particle_coupling_token):
                self.assertNotIn(particle_coupling_token, active_face_flux)


class GasTimeIntegrationSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.dispatch = function_block(cls.source, "advanceGasFluxStage")
        cls.euler_substage = function_block(
            cls.source, "advanceGasEulerSubstage"
        )

    def test_euler_selector_is_a_single_gas_substage(self) -> None:
        euler_branch = branch_block(
            self.dispatch,
            r"if\s*\(\s*s->hostGasTimeIntegrator\s*==\s*1\s*\)",
        )
        self.assertEqual(euler_branch.count("advanceGasEulerSubstage"), 1)
        self.assertIn("return advanceGasEulerSubstage(s, dt);", euler_branch)
        self.assertNotIn("saveGasConservativeStateKernel", euler_branch)

    def test_ssprk2_and_ssprk3_use_gas_conservative_stage_storage(self) -> None:
        active_dispatch = code_only(self.dispatch)
        self.assertIn("saveGasConservativeStateKernel", active_dispatch)
        self.assertEqual(
            active_dispatch.count("advanceGasEulerSubstage"),
            4,
            "one Euler call site plus the two/three SSPRK call sites are required",
        )
        self.assertIn(
            "blendGasRungeKuttaStage(s, 0.5, 0.5)", active_dispatch
        )
        self.assertIn(
            "blendGasRungeKuttaStage(s, 0.75, 0.25)", active_dispatch
        )
        self.assertRegex(
            active_dispatch,
            r"blendGasRungeKuttaStage\s*\(\s*s\s*,\s*"
            r"1\.0\s*/\s*3\.0\s*,\s*2\.0\s*/\s*3\.0\s*\)",
        )

    def test_gas_runge_kutta_stages_do_not_advance_particles(self) -> None:
        gas_stage_code = code_only(self.dispatch + "\n" + self.euler_substage)
        forbidden = (
            r"\b(?:inject|track|relax|sample|correct|bin|compact)"
            r"[A-Za-z0-9_]*Particle",
            r"\bapplyCollisionalPressureKick\b",
            r"\bcomputeGasParticleCoupling\b",
            r"\badvanceParticle",
        )
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, gas_stage_code))


if __name__ == "__main__":
    unittest.main()
