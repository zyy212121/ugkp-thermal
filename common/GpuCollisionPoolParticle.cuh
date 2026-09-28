#pragma once
// One selected-particle collision moment operation; theta storage and read timing are adapters.
template<bool PoissonMode, bool LateThetaAndRng = false>
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
        rng = s.pRng[i];
        if (uniform01Device(rng) >= collisionProbability)
        {
            s.pRng[i] = rng;
            return;
        }
        if constexpr (!LateThetaAndRng) s.pRng[i] = rng;
    }

    // Poisson rejection needs only status, cell and RNG. Read thermal data
    // only for accepted particles; non-Poisson cutoff still precedes physics.
#if GPU_POOL_THETA_AFTER_REJECTION
    theta = GPU_POOL_PARTICLE_THETA(s, i);
    if (!PoissonMode && theta <= GPU_OPERATOR_R(10.0)*s.thetaMin)
    {
        return;
    }

#endif
    const GPU_OPERATOR_REAL m = clampMin(finiteOr(s.pm[i], GPU_POOL_MASS_FALLBACK), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ux = finiteOr(s.pux[i], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uy = finiteOr(s.puy[i], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uz = finiteOr(s.puz[i], GPU_OPERATOR_R(0.0));
    if constexpr (LateThetaAndRng) theta = GPU_POOL_PARTICLE_THETA(s, i);
    const GPU_OPERATOR_REAL d =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );
    const GPU_OPERATOR_REAL specificEnergy =
        GPU_OPERATOR_R(0.5)*sqr3(ux, uy, uz) + GPU_OPERATOR_R(1.5)*theta;

    if
    (
        nonFiniteDevice(m) || m < GPU_OPERATOR_R(0.0)
     || nonFiniteDevice(ux) || nonFiniteDevice(uy)
     || nonFiniteDevice(uz)
     || nonFiniteDevice(theta) || theta < GPU_OPERATOR_R(0.0)
     || nonFiniteDevice(specificEnergy)
    )
    {
        asm("trap;");
    }

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
