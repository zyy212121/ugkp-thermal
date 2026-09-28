#pragma once
// Shared Gaussian sampling and weighted moments; native contact metadata remains an extension.
__device__ void sampleOnePoissonPoolParticle
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
        s.pStatus[i] = 0;
        return;
    }

    const GPU_OPERATOR_REAL invM = GPU_OPERATOR_R(1.0)/poolMass;
    const GPU_OPERATOR_REAL targetUx = s.poissonPoolMomX[c]*invM;
    const GPU_OPERATOR_REAL targetUy = s.poissonPoolMomY[c]*invM;
    const GPU_OPERATOR_REAL targetUz = s.poissonPoolMomZ[c]*invM;
    const GPU_OPERATOR_REAL targetKinetic = GPU_OPERATOR_R(0.5)*sqr3(targetUx, targetUy, targetUz);
    const GPU_OPERATOR_REAL targetThetaRaw =
        clampMin(s.poissonPoolEnergy[c]*invM - targetKinetic, GPU_OPERATOR_R(0.0))/GPU_OPERATOR_R(1.5);

    GPU_OPERATOR_REAL targetTheta = targetThetaRaw;

    if (applyThetaDrag)
    {
        const GPU_OPERATOR_REAL alphaTheta =
            clampRange(finiteOr(s.thetaDragAlpha[c], GPU_OPERATOR_R(1.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));

        targetTheta *= alphaTheta;
    }

    const GPU_OPERATOR_REAL sigma = sqrt(clampRange(targetTheta, GPU_OPERATOR_R(0.0), OfGreat));
    unsigned long long rng = s.pRng[i];

    GPU_OPERATOR_REAL z0 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL z1 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL z2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL z3 = GPU_OPERATOR_R(0.0);

    normalPairDevice(rng, z0, z1);
    normalPairDevice(rng, z2, z3);
    (void) z3;

    const GPU_OPERATOR_REAL ux = targetUx + sigma*z0;
    const GPU_OPERATOR_REAL uy = targetUy + sigma*z1;
    const GPU_OPERATOR_REAL uz = targetUz + sigma*z2;
    if
    (
        n > 1
     && (
            nonFiniteDevice(ux) || nonFiniteDevice(uy) || nonFiniteDevice(uz)
        )
    )
    {
        asm("trap;");
    }

#if GPU_OPERATOR_THERMAL
    const bool wasStuck = s.pStuck[i] != 0;
    const bool wasFiniteContact =
        s.pStuck[i] == Foam::gpuThermal::particleWallTransientRebound
     || s.pStuck[i] == Foam::gpuThermal::particleWallTransientDeposit;
    const GPU_OPERATOR_TIME finiteContactAge = wasFiniteContact ? GPU_CONTACT_AGE(s, i) : GPU_CONTACT_TIME_ZERO;
#endif
    s.pux[i] = finiteOr(ux, targetUx);
    s.puy[i] = finiteOr(uy, targetUy);
    s.puz[i] = finiteOr(uz, targetUz);
    s.pTheta[i] = GPU_OPERATOR_R(0.0);
    GPU_RESET_CONTACT_AGE(s, i)
    const GPU_OPERATOR_REAL dMin = clampMin(s.particleDiameterMin, GPU_OPERATOR_R(1.0e-12));
    const GPU_OPERATOR_REAL dMax = clampMin(s.particleDiameterMax, dMin);

    s.pd[i] =
        clampRange
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            dMin,
            dMax
        );

#if GPU_OPERATOR_THERMAL
    if (!finiteDevice(s.pm[i]) || s.pm[i] <= GPU_OPERATOR_R(0.0))
    {
        asm("trap;");
    }
    if (!wasStuck)
    {
        s.pStuckFaceId[i] = -1;
        s.pDepositionArea[i] = 0.0f;
        s.pContactDuration[i] = 0.0f;
        s.pContactMaximumArea[i] = 0.0f;
        s.pContactPeakFraction[i] = 0.0f;
    }
    else if (wasFiniteContact)
    {
        GPU_CONTACT_AGE(s, i) = finiteContactAge;
    }
#else
    const GPU_OPERATOR_REAL sampleMass = clampMin(finiteOr(s.pm[i], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (!(sampleMass > GPU_OPERATOR_R(0.0)))
    {
        asm("trap;");
        return;
    }
#endif
    s.pRng[i] = rng;

#if GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL sampleMass = s.pm[i];
#endif
    const GPU_OPERATOR_REAL sampleFluctuationX = s.pux[i] - targetUx;
    const GPU_OPERATOR_REAL sampleFluctuationY = s.puy[i] - targetUy;
    const GPU_OPERATOR_REAL sampleFluctuationZ = s.puz[i] - targetUz;
    atomicAdd
    (
        &s.poolThermalSumUx[c],
        sampleMass*sampleFluctuationX
    );
    atomicAdd
    (
        &s.poolThermalSumUy[c],
        sampleMass*sampleFluctuationY
    );
    atomicAdd
    (
        &s.poolThermalSumUz[c],
        sampleMass*sampleFluctuationZ
    );
    atomicAdd
    (
        &s.poolThermalSumU2[c],
        sampleMass*sqr3
        (
            sampleFluctuationX,
            sampleFluctuationY,
            sampleFluctuationZ
        )
    );
#if GPU_OPERATOR_THERMAL
    appendSelectedStuckParticleIndex(s, i);
#endif
}
