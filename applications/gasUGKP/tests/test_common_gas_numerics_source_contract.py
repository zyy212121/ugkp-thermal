from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class CommonGasNumericsSourceContract(unittest.TestCase):
    def assert_links(self, canonical, paths):
        source = ROOT / canonical
        self.assertTrue(source.is_file())
        self.assertFalse(source.is_symlink())
        for relative in paths:
            link = ROOT / relative
            self.assertTrue(link.is_symlink(), relative)
            self.assertEqual(link.resolve(), source.resolve(), relative)

    def test_three_solver_cuda_sources_have_one_entity(self):
        roots = (
            "applications/gasUGKP/private_backend",
            "applications/FSH/private_backend",
            "applications/CHT/gpu",
        )
        for name in (
            "RiemannGasFlux.cuh",
            "RiemannBoundaryState.cuh",
            "CharacteristicMuscl.cuh",
            "OpenFoamViscousFlux.cuh",
            "OpenFoamLimitedLinear.cuh",
            "OpenFoamWallFunctions.cuh",
            "GpuSstAlgebra.cuh",
        ):
            self.assert_links(
                f"common/gasNumerics/{name}",
                tuple(f"{root}/{name}" for root in roots),
            )

    def test_shared_subset_cuda_sources_have_one_entity(self):
        self.assert_links(
            "common/gasNumerics/FullSecondOrderGks.cuh",
            (
                "applications/gasUGKP/private_backend/FullSecondOrderGks.cuh",
                "applications/FSH/private_backend/FullSecondOrderGks.cuh",
            ),
        )
        self.assert_links(
            "common/gasNumerics/GpuDragModelsFSHCHT.cuh",
            (
                "applications/FSH/private_backend/GpuDragModels.cuh",
                "applications/CHT/gpu/GpuDragModels.cuh",
            ),
        )
        gas_drag = ROOT / "applications/gasUGKP/private_backend/GpuDragModels.cuh"
        self.assertTrue(gas_drag.is_file())
        self.assertFalse(gas_drag.is_symlink())
        self.assertNotEqual(
            gas_drag.read_bytes(),
            (ROOT / "common/gasNumerics/GpuDragModelsFSHCHT.cuh").read_bytes(),
        )

    def test_common_particle_drag_and_les_algebra_have_one_owner(self):
        canonical_files = (
            "common/gasNumerics/GpuDragAlgebra.cuh",
            "common/gasNumerics/GpuLesAlgebra.cuh",
            "common/gasNumerics/GpuPackingProjectionAlgebra.cuh",
            "common/gasNumerics/GpuParticlePhysicsAlgebra.cuh",
        )
        for relative in canonical_files:
            source = ROOT / relative
            self.assertTrue(source.is_file(), relative)
            self.assertFalse(source.is_symlink(), relative)

        self.assert_links(
            "common/gasNumerics/GpuDragAlgebra.cuh",
            (
                "applications/gasUGKP/private_backend/GpuDragAlgebra.cuh",
                "applications/FSH/private_backend/GpuDragAlgebra.cuh",
                "applications/CHT/gpu/GpuDragAlgebra.cuh",
            ),
        )

        backends = (
            ROOT / "applications/gasUGKP/private_backend/GpuResidentStrict.cu",
            ROOT / "applications/FSH/private_backend/GpuResidentStrict.cu",
            ROOT / "applications/CHT/gpu/GpuResidentStrict.cu",
        )
        for backend in backends:
            text = backend.read_text(encoding="utf-8")
            self.assertIn("GpuLesAlgebra.cuh", text)
            self.assertIn("GpuPackingProjectionAlgebra.cuh", text)
            self.assertIn("GpuParticlePhysicsAlgebra.cuh", text)
            self.assertNotIn("0.6*sqrt(clampMin(re, 0.0))", text)
            self.assertNotIn("2.0*(1.0 + s.collisionalRestitution)", text)
            self.assertNotIn("__device__ void mobilePackingPrimitive", text)
            self.assertNotIn(
                "__device__ double mobilePackingProjectedJacobiValue",
                text,
            )
            self.assertNotIn(
                "__device__ void mobilePackingParticleVelocityCorrection",
                text,
            )

        gas_drag = (
            ROOT / "applications/gasUGKP/private_backend/GpuDragModels.cuh"
        ).read_text(encoding="utf-8")
        shared_drag = (
            ROOT / "common/gasNumerics/GpuDragModelsFSHCHT.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn("GpuDragAlgebra.cuh", gas_drag)
        self.assertIn("GpuDragAlgebra.cuh", shared_drag)
        self.assertNotIn("pow(reSafe, 0.687)", gas_drag)
        self.assertNotIn("pow(alphaRe, 0.687)", shared_drag)

    def test_three_solver_host_interfaces_have_one_entity(self):
        roots = (
            "applications/gasUGKP/gpu",
            "applications/FSH/gpu",
            "applications/CHT/gpu",
        )
        for name in (
            "GpuBoundarySchedule.H",
            "GpuCouplingMath.H",
            "GpuGasBoundarySpec.H",
        ):
            self.assert_links(
                f"common/gpu/{name}",
                tuple(f"{root}/{name}" for root in roots),
            )


if __name__ == "__main__":
    unittest.main()
