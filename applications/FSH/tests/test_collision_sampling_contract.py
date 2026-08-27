#!/usr/bin/env python3
                                                                 

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CollisionSamplingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text(
            encoding="utf-8", errors="strict"
        )

    def test_only_first_poisson_collision_pool_is_sampled(self) -> None:
        self.assertNotIn(
            "accumulateUnresolvedThetaPoolParticlesByCellKernel", self.cuda
        )
        self.assertNotIn("ProbeThetaPool", self.cuda)
        self.assertNotIn("materialized theta pool", self.cuda)

    def test_collision_sampling_preserves_particle_temperature(self) -> None:
        start = self.cuda.index("__device__ void sampleOnePoissonPoolParticle")
        end = self.cuda.index("__device__ void finalizeOneThermalizedParticle", start)
        self.assertNotIn("s.pT[i] =", self.cuda[start:end])
        self.assertNotIn("poissonPoolHeat", self.cuda)

    def test_unresolved_theta_keeps_drag_energy_balance_without_sampling(self) -> None:
        start = self.cuda.index("__device__ void relaxOneParticleToResidentGas")
        end = self.cuda.index("__global__ void relaxMobileParticlesToResidentGasKernelStatic", start)
        body = self.cuda[start:end]
        self.assertIn("finiteOr(s.thetaDragAlpha[c], 1.0)", body)
        self.assertIn("s.pTheta[i] = unresolvedTheta*alphaTheta;", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
