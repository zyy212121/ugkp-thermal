// Intentionally included twice: one maintained traversal, with no call boundary.
    GPU_MOMENT_REAL rho = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL momX = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL momY = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL momZ = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL energy = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL diameter = GPU_MOMENT_R(0.0);
    GPU_MOMENT_REAL heat = GPU_MOMENT_R(0.0);
    GPU_MOMENT_COUNT_TYPE GPU_MOMENT_COUNT = GPU_MOMENT_R(0.0);

    #if !GPU_MOMENT_THERMAL
    const GPU_MOMENT_REAL heatFactor = particleHeatFactorDevice(s);
#endif
    for (int pos = GPU_MOMENT_BEGIN + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];
        if
        (
            i < 0
         || i >= s.particleCapacity
         || s.pStatus[i] != 1
         || s.pCellId[i] != c
        )
        {
            continue;
        }

#include "GpuParticleMomentContribution.inl"
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += particleEnergy;
        diameter += m*d;
#if GPU_MOMENT_THERMAL
        heat += particleHeat;
#else
        if (heatFactor > GPU_MOMENT_R(0.0)) heat += particleHeat;
#endif
        #if GPU_MOMENT_GATHER
        if constexpr (GatherSurvivors) copyCellLocalParticle(s, i, c, pos);
#endif
        ++GPU_MOMENT_COUNT;
    }

