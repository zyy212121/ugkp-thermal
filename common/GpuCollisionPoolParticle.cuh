#pragma once
#include "GpuPoolSelection.cuh"
// One selected-particle operation and Poisson access schedule for every app/hierarchy.
// A directory assigns each particle to exactly one lane. Accepted RNG/status
// stores follow contribution loads; rejected draws are still committed immediately.
// The explicit false specialization remains usable for equivalence tests.
template<bool PoissonMode, bool LateThetaAndRng = PoissonMode>
__device__ __forceinline__ void accumulateOnePoolParticle
(
    DeviceState& s,
    const int c,
    const int i,
    const GPU_OPERATOR_REAL collisionProbability,
    GPU_OPERATOR_REAL& mass,
    GPU_OPERATOR_REAL& momX,
    GPU_OPERATOR_REAL& momY,
    GPU_OPERATOR_REAL& momZ,
    GPU_OPERATOR_REAL& energy,
    GPU_OPERATOR_REAL& diameter,
    GPU_OPERATOR_REAL& diameter2,
    GPU_OPERATOR_REAL& count
)
{
    static_assert(!LateThetaAndRng || PoissonMode, "Late theta requires Poisson selection");
    if
    (
        i < 0
     || i >= s.particleCapacity
     || s.pStatus[i] == 0
     || s.pCellId[i] != c
    )
    {
        return;
    }

    GPU_OPERATOR_REAL theta;
#if !GPU_POOL_THETA_AFTER_REJECTION
    if constexpr (!LateThetaAndRng)
    {
    theta = GPU_POOL_PARTICLE_THETA(s, i);
    if (!PoissonMode && theta <= GPU_OPERATOR_R(10.0)*s.thetaMin)
    {
        return;
    }

    }
#endif
    unsigned long long rng;
    if (PoissonMode)
    {
        if (!selectPoissonPoolParticle<LateThetaAndRng>(s, i, collisionProbability, rng))
        {
            return;
        }
    }

    // Poisson rejection needs only status, cell and RNG. Read thermal data
    // only for accepted particles; non-Poisson cutoff still precedes physics.
#if GPU_POOL_THETA_AFTER_REJECTION
    if constexpr (!LateThetaAndRng)
    {
        theta = GPU_POOL_PARTICLE_THETA(s, i);
        if (!PoissonMode && theta <= GPU_OPERATOR_R(10.0)*s.thetaMin)
        {
            return;
        }
    }

#endif
#define GPU_POOL_LOAD_LATE_THETA() if constexpr (LateThetaAndRng) theta = GPU_POOL_PARTICLE_THETA(s, i);
#include "GpuPoolParticleContribution.inl"
#undef GPU_POOL_LOAD_LATE_THETA
    mass += m;
    momX += m*ux;
    momY += m*uy;
    momZ += m*uz;
    energy += m*specificEnergy;
    diameter += m*d;
    diameter2 += m*d*d;
    count += GPU_OPERATOR_R(1.0);
    if constexpr (PoissonMode && LateThetaAndRng) s.pRng[i] = rng;
    s.pStatus[i] = 2;
}
