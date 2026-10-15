#!/usr/bin/env python3
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

                                                                 

from source_contract_utils import read_source, function_block
from pathlib import Path
import unittest


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')


class CollisionSamplingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = read_source(ROOT / "private_backend/GpuResidentStrict.cu")

    def test_only_first_poisson_collision_pool_is_sampled(self) -> None:
        self.assertNotIn(
            "accumulateUnresolvedThetaPoolParticlesByCellKernel", self.cuda
        )
        self.assertNotIn("ProbeThetaPool", self.cuda)
        self.assertNotIn("materialized theta pool", self.cuda)

    def test_collision_sampling_preserves_particle_temperature(self) -> None:
        start = self.cuda.index("__device__ void sampleOnePoissonPoolParticle")
        end = self.cuda.index("__global__ void samplePoissonPoolParticlesKernel", start)
        self.assertNotIn("s.pT[i] =", self.cuda[start:end])
        self.assertNotIn("poissonPoolHeat", self.cuda)

    def test_unresolved_theta_keeps_drag_energy_balance_without_sampling(self) -> None:
        start = self.cuda.index("__device__ void relaxOneParticleToResidentGas")
        end = self.cuda.index("__global__ void relaxParticlesToResidentGasKernel", start)
        body = self.cuda[start:end]
        self.assertIn("decayParticleUnresolvedTheta(s, i, c)", body)
        decay = function_block(self.cuda, "decayParticleUnresolvedTheta")
        self.assertIn("finiteOr(s.thetaDragAlpha[c], 1.0)", decay)
        self.assertIn("s.pTheta[i] = theta*alpha;", decay)

    def test_collision_sampling_preserves_heterogeneous_particle_mass(self) -> None:
        start = self.cuda.index("__device__ void sampleOnePoissonPoolParticle")
        end = self.cuda.index("__global__ void samplePoissonPoolParticlesKernel", start)
        body = self.cuda[start:end]
        self.assertIn("const double sampleMass =", body)
        self.assertIn("const double sampleFluctuationX = s.pux[i] - targetUx;", body)
        self.assertIn("const double sampleFluctuationY = s.puy[i] - targetUy;", body)
        self.assertIn("const double sampleFluctuationZ = s.puz[i] - targetUz;", body)
        self.assertIn("sampleMass*sampleFluctuationX", body)
        self.assertIn("sampleMass*sampleFluctuationY", body)
        self.assertIn("sampleMass*sampleFluctuationZ", body)
        self.assertIn(
            "sampleMass*sqr3\n        (\n            sampleFluctuationX,\n            sampleFluctuationY,\n            sampleFluctuationZ\n        )", body
        )
        self.assertNotIn("s.pm[i] =", body)

    def test_collision_correction_uses_mass_weighted_affine_projection(self) -> None:
        start = self.cuda.index("__device__ void correctOnePoissonThermalizedParticle")
        end = self.cuda.index("__global__ void correctPoissonThermalizedParticlesKernel", start)
        body = self.cuda[start:end]
        self.assertIn(
            "sampleMeanDeltaX = s.poolThermalSumUx[c]/poolMass", body
        )
        self.assertIn(
            "(s.pux[i] - targetUx) - sampleMeanDeltaX", body
        )
        self.assertIn("1.5*poolMass*targetTheta", body)
        self.assertIn("const double meanParticleMass =", body)
        self.assertIn("poolMass/static_cast<double>(n);", body)
        self.assertIn("sampleFluctuationEnergy <= OfSmall*meanParticleMass", body)
        self.assertNotIn("sampleFluctuationEnergy <= OfSmall)", body)
        self.assertIn("s.pTheta[i] = targetTheta;", body)
        self.assertNotIn("const double invN", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
