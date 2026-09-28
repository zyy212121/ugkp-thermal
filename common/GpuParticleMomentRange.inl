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

        const GPU_MOMENT_REAL m = clampMin(finiteOr(s.pm[i], GPU_MOMENT_R(0.0)), GPU_MOMENT_R(0.0));
        const GPU_MOMENT_REAL ux = finiteOr(s.pux[i], GPU_MOMENT_R(0.0));
        const GPU_MOMENT_REAL uy = finiteOr(s.puy[i], GPU_MOMENT_R(0.0));
        const GPU_MOMENT_REAL uz = finiteOr(s.puz[i], GPU_MOMENT_R(0.0));
        #if GPU_MOMENT_THERMAL
        const GPU_MOMENT_REAL theta = particleMomentThetaDevice(s, i);
        const GPU_MOMENT_REAL d = clampMin(finiteOr(s.pd[i], s.particleDiameterFallback), GPU_MOMENT_R(1.0e-12));
        const GPU_MOMENT_REAL tp = clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
#else
        const GPU_MOMENT_REAL theta = clampMin(finiteOr(s.pTheta[i], GPU_MOMENT_R(0.0)), GPU_MOMENT_R(0.0));
#endif
        rho += m;
        momX += m*ux;
        momY += m*uy;
        momZ += m*uz;
        energy += m*(GPU_MOMENT_R(0.5)*sqr3(ux, uy, uz) + GPU_MOMENT_R(1.5)*theta);
#if GPU_MOMENT_THERMAL
        diameter += m*d;
        heat += m*particleSpecificEnthalpyDevice(tp);
#else
        diameter += m*clampMin(finiteOr(s.pd[i], s.particleDiameterFallback), GPU_MOMENT_R(1.0e-12));
        if (heatFactor > GPU_MOMENT_R(0.0))
        {
            heat += m*heatFactor
                *clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
        }
#endif
        #if GPU_MOMENT_GATHER
        if constexpr (GatherSurvivors) copyCellLocalParticle(s, i, c, pos);
#endif
        ++GPU_MOMENT_COUNT;
    }

