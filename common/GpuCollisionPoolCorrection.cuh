#pragma once
// Shared weighted sampling correction; physical contact finalization is an extension.
#if GPU_OPERATOR_THERMAL
template<bool StuckPath>
__device__ void correctOnePoissonThermalizedParticlePath
#else
__device__ void correctOnePoissonThermalizedParticle
#endif
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    if (i >= s.particleCapacity || s.pStatus[i] != 2)
    {
        return;
    }
#if GPU_OPERATOR_THERMAL
    if (StuckPath)
    {
        if (s.pStuck[i] == 0)
        {
            asm("trap;");
        }
    }
    else if (s.pStuck[i] != 0)
    {
        return;
    }

#endif
    const int c = s.pCellId[i];
    if (c < 0 || c >= s.nCells)
    {
        s.pStatus[i] = 0;
        return;
    }

    const int n = s.poissonPoolSampleTargetCount[c];
    const GPU_OPERATOR_REAL poolMass = clampMin(s.poissonPoolMass[c], GPU_OPERATOR_R(0.0));
    if (n <= 0 || poolMass <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

#if !GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL meanParticleMass = poolMass/static_cast<GPU_OPERATOR_REAL>(n);
#endif
    const GPU_OPERATOR_REAL targetUx = s.poissonPoolMomX[c]/poolMass;
    const GPU_OPERATOR_REAL targetUy = s.poissonPoolMomY[c]/poolMass;
    const GPU_OPERATOR_REAL targetUz = s.poissonPoolMomZ[c]/poolMass;
    const GPU_OPERATOR_REAL targetMean2 = sqr3(targetUx, targetUy, targetUz);
    const GPU_OPERATOR_REAL targetThetaRaw =
        clampMin(s.poissonPoolEnergy[c]/poolMass - GPU_OPERATOR_R(0.5)*targetMean2, GPU_OPERATOR_R(0.0))/GPU_OPERATOR_R(1.5);
#if GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL meanParticleMass = poolMass/static_cast<GPU_OPERATOR_REAL>(n);
#endif
    const GPU_OPERATOR_REAL sampleMeanDeltaX = s.poolThermalSumUx[c]/poolMass;
    const GPU_OPERATOR_REAL sampleMeanDeltaY = s.poolThermalSumUy[c]/poolMass;
    const GPU_OPERATOR_REAL sampleMeanDeltaZ = s.poolThermalSumUz[c]/poolMass;
    const GPU_OPERATOR_REAL sampleMeanDelta2 = sqr3
    (
        sampleMeanDeltaX,
        sampleMeanDeltaY,
        sampleMeanDeltaZ
    );
    const GPU_OPERATOR_REAL sampleFluctuationEnergy = GPU_OPERATOR_R(0.5)*clampMin
    (
        s.poolThermalSumU2[c] - poolMass*sampleMeanDelta2,
        GPU_OPERATOR_R(0.0)
    );

    GPU_OPERATOR_REAL targetTheta = targetThetaRaw;

    if (applyThetaDrag)
    {
        const GPU_OPERATOR_REAL alphaTheta =
            clampRange(finiteOr(s.thetaDragAlpha[c], GPU_OPERATOR_R(1.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));

        targetTheta *= alphaTheta;
    }

    const GPU_OPERATOR_REAL targetRandomEnergy = GPU_OPERATOR_R(1.5)*poolMass*targetTheta;

    if (targetRandomEnergy <= GPU_OPERATOR_R(0.0))
    {
#if GPU_OPERATOR_THERMAL
        finalizeOneThermalizedParticlePath<StuckPath>
        (
            s, i,
            targetUx, targetUy, targetUz,
            targetUx, targetUy, targetUz
        );
#else
        s.pux[i] = targetUx;
        s.puy[i] = targetUy;
        s.puz[i] = targetUz;
        s.pTheta[i] = GPU_OPERATOR_R(0.0);
        s.pStatus[i] = 1;
#endif
        return;
    }

#if GPU_OPERATOR_THERMAL
    const bool retainSingletonTheta =
        n <= 1
     || sampleFluctuationEnergy <= OfSmall*meanParticleMass;
    if (retainSingletonTheta)
    {
        if (StuckPath)
        {
            finalizeOneThermalizedParticlePath<true>
            (
                s, i,
                s.pux[i], s.puy[i], s.puz[i],
                targetUx, targetUy, targetUz
            );
        }
        else
        {
            s.pux[i] = targetUx;
            s.puy[i] = targetUy;
            s.puz[i] = targetUz;
            s.pTheta[i] = targetTheta;
            s.pStatus[i] = 1;
        }
        return;
    }

#else
    if (n <= 1 || sampleFluctuationEnergy <= OfSmall*meanParticleMass)
    {
                                                                            
                                                                       
                                                                            
        s.pux[i] = targetUx;
        s.puy[i] = targetUy;
        s.puz[i] = targetUz;
        s.pTheta[i] = targetTheta;
        s.pStatus[i] = 1;
        return;
    }

#endif
    GPU_OPERATOR_REAL scale = GPU_OPERATOR_R(1.0);
    scale = sqrt(targetRandomEnergy/sampleFluctuationEnergy);
    const GPU_OPERATOR_REAL correctedUx = targetUx + scale*
        ((s.pux[i] - targetUx) - sampleMeanDeltaX);
    const GPU_OPERATOR_REAL correctedUy = targetUy + scale*
        ((s.puy[i] - targetUy) - sampleMeanDeltaY);
    const GPU_OPERATOR_REAL correctedUz = targetUz + scale*
        ((s.puz[i] - targetUz) - sampleMeanDeltaZ);
    if
    (
        nonFiniteDevice(correctedUx) || nonFiniteDevice(correctedUy)
     || nonFiniteDevice(correctedUz)
    )
    {
        asm("trap;");
    }
#if GPU_OPERATOR_THERMAL
    finalizeOneThermalizedParticlePath<StuckPath>
    (
        s, i,
        correctedUx, correctedUy, correctedUz,
        targetUx, targetUy, targetUz
    );
#else
    s.pux[i] = correctedUx;
    s.puy[i] = correctedUy;
    s.puz[i] = correctedUz;
    s.pTheta[i] = GPU_OPERATOR_R(0.0);
    s.pStatus[i] = 1;
#endif
}
