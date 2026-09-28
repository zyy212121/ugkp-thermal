#pragma once
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

        const GPU_OPERATOR_REAL theta = GPU_POOL_PARTICLE_THETA(s, i);
        if (!PoissonMode && theta <= GPU_OPERATOR_R(10.0)*s.thetaMin)
        {
            continue;
        }
        if (PoissonMode)
        {
#if GPU_OPERATOR_THERMAL
            const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);
            const GPU_OPERATOR_REAL probability =
                (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
              ? GPU_OPERATOR_R(0.0)
              : clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
#else
            const GPU_OPERATOR_REAL probability = poissonCollisionProbabilityForCell(s, c, dt);
#endif
            if (probability <= GPU_OPERATOR_R(0.0))
            {
                continue;
            }
            unsigned long long rng = s.pRng[i];
            if (uniform01Device(rng) >= probability)
            {
                s.pRng[i] = rng;
                continue;
            }
            s.pRng[i] = rng;
        }

        const GPU_OPERATOR_REAL m =
            clampMin(finiteOr(s.pm[i], GPU_POOL_MASS_FALLBACK), GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL ux = finiteOr(s.pux[i], GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL uy = finiteOr(s.puy[i], GPU_OPERATOR_R(0.0));
        const GPU_OPERATOR_REAL uz = finiteOr(s.puz[i], GPU_OPERATOR_R(0.0));
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
