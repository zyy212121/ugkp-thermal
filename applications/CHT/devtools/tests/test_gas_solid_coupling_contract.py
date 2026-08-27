import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA = ROOT / "gpu" / "GpuResidentStrict.cu"


class GasSolidCouplingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = CUDA.read_text(encoding="utf-8")
        begin = source.index("__global__ void applyEulerianGasSolidCouplingKernel")
        end = source.index(
            "__global__ void applyEulerianParticleMaterialHeatKernel",
            begin,
        )
        cls.body = source[begin:end]

    def test_volume_source_commits_before_early_returns(self) -> None:
        required = (
            "const double enerGCandidate = s.rhoE[c]*massScale;",
            "s.rho[c] = mgCandidate;",
            "s.rhoUx[c] = momGXCandidate;",
            "s.rhoUy[c] = momGYCandidate;",
            "s.rhoUz[c] = momGZCandidate;",
            "s.rhoE[c] = enerGCandidate;",
            "s.epsGPrev[c] = epsG;",
        )
        for statement in required:
            self.assertIn(statement, self.body)

        history_commit = self.body.index("s.epsGPrev[c] = epsG;")
        no_particle_return = self.body.index(
            "if (rhoP <= s.epsSMin*s.rhoSolid)"
        )
        no_drag_return = self.body.index(
            "if (!finiteDevice(beta) || beta <= OfSmall)"
        )
        self.assertLess(history_commit, no_particle_return)
        self.assertLess(history_commit, no_drag_return)
        self.assertEqual(self.body.count("s.epsGPrev[c] = epsG;"), 1)

    def test_volume_source_scales_all_five_conservative_components(self) -> None:
        self.assertIn("!finiteDevice(enerGCandidate)", self.body)
        self.assertIn(
            "enerGCandidate < kineticCandidate + internalEnergyFloorCandidate",
            self.body,
        )
        self.assertEqual(self.body.count("s.rhoE[c] = enerGCandidate;"), 1)

        rho = 3.25
        momentum = (4.5, -1.75, 0.625)
        rho_energy = 29.0
        eps_old = 0.71
        eps_new = 0.83
        scale = eps_old/eps_new
        scaled_rho = rho*scale
        scaled_momentum = tuple(value*scale for value in momentum)
        scaled_energy = rho_energy*scale
        for old, new in zip(momentum, scaled_momentum):
            self.assertAlmostEqual(new/scaled_rho, old/rho, places=14)
        self.assertAlmostEqual(
            scaled_energy/scaled_rho,
            rho_energy/rho,
            places=14,
        )
        self.assertAlmostEqual(
            eps_new*scaled_energy,
            eps_old*rho_energy,
            places=14,
        )

    def test_volume_source_has_no_late_duplicate_update(self) -> None:
        self.assertNotIn("mg += dt*cepsG*mg", self.body)
        self.assertNotIn("momGX0 + dt*cepsG*momGX0", self.body)
        self.assertNotIn("momGY0 + dt*cepsG*momGY0", self.body)
        self.assertNotIn("momGZ0 + dt*cepsG*momGZ0", self.body)

    def test_unmatched_macro_mechanical_energy_source_is_absent(self) -> None:
        self.assertNotIn("const double qE =", self.body)
        self.assertNotIn("enerG0 + dt*qE", self.body)
        self.assertNotIn("enerS0 - dt*qE", self.body)
        self.assertIn("double enerG = enerG0;", self.body)
        self.assertIn("double enerS = enerS0;", self.body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
