#pragma once
// Shared Gaussian sampling and weighted moments; native contact metadata remains an extension.
template<bool AtomicMoments = true>
__device__ void sampleOnePoissonPoolParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag,
    GPU_OPERATOR_REAL* moments = nullptr
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

    const GPU_OPERATOR_REAL targetUx = s.poissonPoolMomX[c];
    const GPU_OPERATOR_REAL targetUy = s.poissonPoolMomY[c];
    const GPU_OPERATOR_REAL targetUz = s.poissonPoolMomZ[c];
    const GPU_OPERATOR_REAL targetThetaRaw = collisionCentralTheta(
        {poolMass,targetUx,targetUy,targetUz,s.poissonPoolEnergy[c]});

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

    const bool singletonMobile = n == 1
#if GPU_OPERATOR_THERMAL
        && s.pStuck[i] == 0
#endif
        ;
    if (singletonMobile)
    {
        // This collision is an identity for a mobile singleton unless the
        // caller requests physical theta drag. Never reconstruct old theta
        // from a rounded M*theta/M, even when alpha is exactly one.
        if (applyThetaDrag)
        {
            const float alpha=clampRange(finiteOr(s.thetaDragAlpha[c],1.0f),0.0f,1.0f);
            if (alpha != 1.0f) s.pTheta[i] *= alpha;
        }
        s.pRng[i]=rng;
        if constexpr (!AtomicMoments)
        {
            const float one[5]={s.pm[i],s.pux[i],s.puy[i],s.puz[i],1.5f*s.pm[i]*s.pTheta[i]};
            // The singleton's unresolved theta is retained on the parcel.
            float state[5]={moments[0],moments[1],moments[2],moments[3],moments[4]};
            mergeCollisionMomentGroup(state,one);
            for(int k=0;k<5;++k)moments[k]=state[k];
        }
        return;
    }

    const GPU_OPERATOR_REAL ux = targetUx + sigma*z0;
    const GPU_OPERATOR_REAL uy = targetUy + sigma*z1;
    const GPU_OPERATOR_REAL uz = targetUz + sigma*z2;

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

#endif
    s.pRng[i] = rng;

#if GPU_OPERATOR_THERMAL
    const GPU_OPERATOR_REAL sampleMass = s.pm[i];
#endif
    if constexpr (!AtomicMoments)
    {
        const float one[5]={sampleMass,s.pux[i],s.puy[i],s.puz[i],0.0f};
        float state[5]={moments[0],moments[1],moments[2],moments[3],moments[4]};
        mergeCollisionMomentGroup(state,one);
        for(int k=0;k<5;++k)moments[k]=state[k];
    }
#if GPU_OPERATOR_THERMAL
    appendSelectedStuckParticleIndex(s, i);
#endif
}
