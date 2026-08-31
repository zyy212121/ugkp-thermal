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
        begin = source.index("__global__ void applyGasVolumeFractionSourceKernel")
        end = source.index("__global__ void applyEulerianGasSolidDragKernel", begin)
        cls.body = source[begin:end]
        advance = source.index('extern "C" int ugkwpGpuResidentStrictAdvance\n')
        schedule_begin = source.index("const bool skipParticlePath", advance)
        schedule_end = source.index(
            "UGKP_DEV_PROBE_ENTER(ProbeInjection)",
            schedule_begin,
        )
        cls.schedule = source[schedule_begin:schedule_end]

    def test_history_advances_only_after_mass_and_momentum_commit(self) -> None:
        rho_commit = self.body.index("s.rho[c] = mgCandidate;")
        momentum_commit = self.body.index("s.rhoUx[c] = momGXCandidate;")
        energy_commit = self.body.index("s.rhoE[c] = enerGCandidate;")
        history_commit = self.body.index("s.epsGPrev[c] = epsG;")
        self.assertLess(rho_commit, history_commit)
        self.assertLess(momentum_commit, history_commit)
        self.assertLess(energy_commit, history_commit)

    def test_no_particle_and_no_drag_paths_cannot_skip_volume_source(self) -> None:
        volume_launch = self.schedule.index("applyGasVolumeFractionSourceKernel")
        drag_gate = self.schedule.index("if (dragActive)")
        self.assertLess(volume_launch, drag_gate)

    def test_volume_source_is_not_applied_twice_on_particle_cells(self) -> None:
        self.assertEqual(self.body.count("s.epsGPrev[c] = epsG;"), 1)
        self.assertEqual(
            self.schedule.count("applyGasVolumeFractionSourceKernel<<<grid, block>>>")
            ,
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
