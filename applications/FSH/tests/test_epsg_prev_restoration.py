import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "private_backend" / "GpuResidentStrict.cu"


class GasVolumeSourceMath(unittest.TestCase):
    def test_particle_enter_leave_cycle_restores_density(self) -> None:
        solid_fraction = 0.0094196
        eps_old = 1.0
        eps_enter = 1.0 - solid_fraction
        enter_scale = 1.0 - (eps_enter - eps_old) / eps_enter
        eps_leave = 1.0
        leave_scale = 1.0 - (eps_leave - eps_enter) / eps_leave
        self.assertAlmostEqual(enter_scale, 1.0 / (1.0 - solid_fraction), places=14)
        self.assertAlmostEqual(leave_scale, 1.0 - solid_fraction, places=14)
        self.assertAlmostEqual(enter_scale * leave_scale, 1.0, places=14)


class GasVolumeSourceOrderingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = CUDA.read_text(encoding="utf-8")
        begin = source.index("__global__ void applyEulerianGasSolidCouplingKernel")
        end = source.index("__global__ void applyEulerianParticleMaterialHeatKernel", begin)
        cls.body = source[begin:end]

    def test_history_advances_only_after_mass_and_momentum_commit(self) -> None:
        rho_commit = self.body.index("s.rho[c] = mgCandidate;")
        momentum_commit = self.body.index("s.rhoUx[c] = momGXCandidate;")
        history_commit = self.body.index("s.epsGPrev[c] = epsG;")
        no_particle_return = self.body.index("if (rhoP <= s.epsSMin*s.rhoSolid)")
        self.assertLess(rho_commit, history_commit)
        self.assertLess(momentum_commit, history_commit)
        self.assertLess(history_commit, no_particle_return)

    def test_no_particle_and_no_drag_paths_cannot_skip_volume_source(self) -> None:
        source_commit = self.body.index("s.rho[c] = mgCandidate;")
        no_particle_return = self.body.index("if (rhoP <= s.epsSMin*s.rhoSolid)")
        no_drag_return = self.body.index("if (!finiteDevice(beta) || beta <= OfSmall)")
        self.assertLess(source_commit, no_particle_return)
        self.assertLess(source_commit, no_drag_return)

    def test_volume_source_is_not_applied_twice_on_particle_cells(self) -> None:
        self.assertNotIn("mg += dt*cepsG*mg", self.body)
        self.assertNotIn("momGX0 + dt*cepsG*momGX0", self.body)
        self.assertEqual(self.body.count("s.epsGPrev[c] = epsG;"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
