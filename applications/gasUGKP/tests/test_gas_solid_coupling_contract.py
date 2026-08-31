import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "private_backend" / "GpuResidentStrict.cu"


class GasSolidCouplingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA.read_text(encoding="utf-8")
        volume_begin = cls.source.index(
            "__global__ void applyGasVolumeFractionSourceKernel"
        )
        drag_begin = cls.source.index(
            "__global__ void applyEulerianGasSolidDragKernel",
            volume_begin,
        )
        heat_begin = cls.source.index(
            "__global__ void applyEulerianParticleMaterialHeatKernel",
            drag_begin,
        )
        snapshot_begin = cls.source.index(
            "__global__ void snapshotParticleGasCouplingStateKernel",
            heat_begin,
        )
        init_begin = cls.source.index("__global__ void initialiseEpsGPrevKernel")
        init_end = cls.source.index(
            "__global__ void initialiseThetaDragAlphaKernel",
            init_begin,
        )
        advance_begin = cls.source.index(
            'extern "C" int ugkwpGpuResidentStrictAdvance\n'
        )
        coupling_begin = cls.source.index(
            "const bool skipParticlePath",
            advance_begin,
        )
        coupling_end = cls.source.index(
            "UGKP_DEV_PROBE_ENTER(ProbeInjection)",
            coupling_begin,
        )
        cls.volume = cls.source[volume_begin:drag_begin]
        cls.drag = cls.source[drag_begin:heat_begin]
        cls.heat = cls.source[heat_begin:snapshot_begin]
        cls.initializer = cls.source[init_begin:init_end]
        cls.schedule = cls.source[coupling_begin:coupling_end]

    def test_volume_source_includes_pressure_work(self) -> None:
        self.assertIn(
            "enerGOld*massScale + dt*cepsG*pressureOld",
            self.volume,
        )
        self.assertIn("s.rhoE[c] = enerGCandidate;", self.volume)
        self.assertIn("s.epsGPrev[c] = epsG;", self.volume)
        rho_energy = 29.0
        pressure = 4.0
        ceps = -0.3
        dt = 0.01
        scale = 1.0 + dt*ceps
        corrected = rho_energy*scale + dt*ceps*pressure
        self.assertNotEqual(corrected, rho_energy*scale)

    def test_drag_uses_apparent_gas_capacity_and_intrinsic_writeback(self) -> None:
        required = (
            "const double mg = epsG*rhoG;",
            "const double momGX0 = epsG*s.rhoUx[c];",
            "const double enerG0 = epsG*s.rhoE[c];",
            "const double kW = dt*invTauDragCell*(1.0 + ms/(mg + OfSmall));",
            "s.rhoUx[c] = rhoG*ugNewX;",
            "s.rhoE[c] = (kgNew + igAfter + diss)/epsGsafe;",
        )
        for statement in required:
            self.assertIn(statement, self.drag)
        eps_g = 0.73
        rho_g = 2.8
        rho_p = 0.64
        ug = 12.0
        up = -3.0
        alpha = 0.41
        mg = eps_g*rho_g
        relative_new = (ug - up)*alpha
        total_momentum = mg*ug + rho_p*up
        up_new = (total_momentum - mg*relative_new)/(mg + rho_p)
        ug_new = up_new + relative_new
        self.assertAlmostEqual(
            mg*ug_new + rho_p*up_new,
            total_momentum,
            places=14,
        )

    def test_heat_uses_apparent_capacity_and_intrinsic_writeback(self) -> None:
        self.assertIn("const double gasCapacity = epsG*rhoG*gasCv;", self.heat)
        self.assertIn("rhoEold - dHp/epsGsafe", self.heat)
        eps_g = 0.61
        rho_energy = 18.0
        particle_enthalpy = 7.0
        exchange = 1.25
        gas_after = rho_energy - exchange/eps_g
        particle_after = particle_enthalpy + exchange
        self.assertAlmostEqual(
            eps_g*gas_after + particle_after,
            eps_g*rho_energy + particle_enthalpy,
            places=14,
        )

    def test_volume_source_is_independent_of_drag_schedule(self) -> None:
        volume_launch = self.schedule.index(
            "applyGasVolumeFractionSourceKernel<<<grid, block>>>"
        )
        drag_gate = self.schedule.index("if (dragActive)")
        drag_launch = self.schedule.index("EulerianGasSolidDrag", drag_gate)
        self.assertLess(volume_launch, drag_gate)
        self.assertLess(drag_gate, drag_launch)
        self.assertEqual(
            self.schedule.count("applyGasVolumeFractionSourceKernel<<<grid, block>>>")
            ,
            1,
        )
        self.assertIn("post-volume-source", self.schedule)

    def test_volume_history_initialization_is_drag_independent(self) -> None:
        self.assertNotIn("dragModel", self.initializer)
        self.assertIn("s.epsGPrev[c] = 1.0 - eps;", self.initializer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
