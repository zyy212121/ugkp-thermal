#pragma once
#include "GpuPoolSelection.cuh"
#include "GpuCollisionProbability.cuh"
// Common atomic pool traversal and publication; native probability/scalar policies remain.
template<bool PoissonMode>
__global__ void accumulateParticlePoolAtomicKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] == 0)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }

        GPU_OPERATOR_REAL theta;
        if constexpr (!PoissonMode)
        {
            theta = GPU_POOL_PARTICLE_THETA(s, i);
            if (theta <= GPU_OPERATOR_R(10.0)*s.thetaMin)
            {
                continue;
            }
        }
        if (PoissonMode)
        {
            const GPU_OPERATOR_REAL probability = poissonCollisionProbabilityForCell(s, c, dt);
            if (probability <= GPU_OPERATOR_R(0.0))
            {
                continue;
            }
            unsigned long long rng;
            if (!selectPoissonPoolParticle<false>(s, i, probability, rng))
            {
                continue;
            }
        }
        if constexpr (PoissonMode)
        {
            theta = GPU_POOL_PARTICLE_THETA(s, i);
        }

#define GPU_POOL_LOAD_LATE_THETA()
#include "GpuPoolParticleContribution.inl"
#undef GPU_POOL_LOAD_LATE_THETA
        atomicAdd(&s.poissonPoolMass[c], m);
        atomicAdd(&s.poissonPoolMomX[c], m*ux);
        atomicAdd(&s.poissonPoolMomY[c], m*uy);
        atomicAdd(&s.poissonPoolMomZ[c], m*uz);
        atomicAdd(&s.poissonPoolEnergy[c], m*specificEnergy);
        atomicAdd(&s.poissonPoolDiameter[c], m*d);
        atomicAdd(&s.poissonPoolDiameter2[c], m*d*d);
        atomicAdd(&s.poolThermalCount[c], 1);
        s.pStatus[i] = 2;
    }
}
