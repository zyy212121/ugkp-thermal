#pragma once
__global__ void injectBoundaryParticlesKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt,
    const GPU_OPERATOR_TIME simulationTime
)
{
    DeviceState& s = *sp;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= s.nBoundarySources)
    {
        return;
    }
#if defined(UGKP_DEVELOPMENT_PROBES) && !GPU_OPERATOR_THERMAL
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = 0;
    }
#endif
    if (s.particleCapacity <= 0 || s.injectionParcelMass <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    const int sourceFace = s.sourceFace[j];
    if (sourceFace < 0 || sourceFace >= s.nFaces)
    {
        return;
    }

    const int sourceCell = s.sourceCell[j];
    if (sourceCell < 0 || sourceCell >= s.nCells)
    {
        return;
    }

    const bool scheduledInletActive =
        scheduledInletFaceDevice(s, sourceFace);
    const GPU_OPERATOR_REAL sourceUx = scheduledInletActive
      ? finiteOr(s.gasBoundaryUx[sourceFace], finiteOr(s.sourceUx[j], GPU_OPERATOR_R(0.0)))
      : finiteOr(s.sourceUx[j], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL sourceUy = scheduledInletActive
      ? finiteOr(s.gasBoundaryUy[sourceFace], finiteOr(s.sourceUy[j], GPU_OPERATOR_R(0.0)))
      : finiteOr(s.sourceUy[j], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL sourceUz = scheduledInletActive
      ? finiteOr(s.gasBoundaryUz[sourceFace], finiteOr(s.sourceUz[j], GPU_OPERATOR_R(0.0)))
      : finiteOr(s.sourceUz[j], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL sourceTheta = scheduledInletActive
      ? clampMin(finiteOr(s.theta[sourceCell], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0))
      : clampMin(finiteOr(s.sourceTheta[j], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL scheduledRate =
        scheduledSolidVolumeFractionDevice(s, simulationTime)
       *s.rhoSolid
       *clampMin
        (
           -(sourceUx*s.Sfx[sourceFace]
           + sourceUy*s.Sfy[sourceFace]
           + sourceUz*s.Sfz[sourceFace]),
            GPU_OPERATOR_R(0.0)
        );
    const GPU_OPERATOR_REAL rate = scheduledInletActive
      ? scheduledRate
      : finiteOr(s.sourceMassRate[j], GPU_OPERATOR_R(0.0));
    if (rate <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    const GPU_OPERATOR_REAL injectionParcelMass = s.injectionParcelMass;
    GPU_OPERATOR_REAL available =
        clampMin(finiteOr(s.sourceResidualMass[j], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0)) + rate*dt;
    if (available < injectionParcelMass)
    {
        s.sourceResidualMass[j] = available;
        return;
    }

    const int nNew = static_cast<int>(floor(available/injectionParcelMass));
    GPU_OPERATOR_REAL consumed = GPU_OPERATOR_R(0.0);
    int created = 0;

    for (int k = 0; k < nNew; ++k)
    {
        const int slot = atomicAdd(s.particleCountDevice, 1);
        if (slot >= s.particleCapacity)
        {
            atomicSub(s.particleCountDevice, 1);
            break;
        }

        unsigned long long rng =
            mixSeed
            (
                s.rngSeed
              ^ static_cast<unsigned long long>
                (
                    0xd1b54a32d192ed03ULL
                  + (static_cast<unsigned long long>(j) << 32)
                  + static_cast<unsigned long long>(slot)
                )
            );
        s.px[slot] = finiteOr(s.sourcePx[j], s.Cx[sourceCell]);
        s.py[slot] = finiteOr(s.sourcePy[j], s.Cy[sourceCell]);
        s.pz[slot] = finiteOr(s.sourcePz[j], s.Cz[sourceCell]);
        s.pux[slot] = sourceUx;
        s.puy[slot] = sourceUy;
        s.puz[slot] = sourceUz;
        s.pT[slot] =
            clampRange(finiteOr(s.sourceT[j], s.TpMin), s.TpMin, s.TpMax);
        s.pTheta[slot] = sourceTheta;
        resetSeparateContactAge(s, slot, 0);
        s.pd[slot] = sampleDiameterAroundDevice(s, finiteOr(s.sourceD[j], s.particleDiameterFallback), rng);
        s.pm[slot] = injectionParcelMass;
        s.pCellId[slot] = sourceCell;
        s.pStatus[slot] = 1;
#if GPU_OPERATOR_THERMAL
        s.pStuck[slot] = 0;
        s.pStuckFaceId[slot] = -1;
        s.pDepositionArea[slot] = 0.0f;
        s.pContactDuration[slot] = 0.0f;
        s.pContactMaximumArea[slot] = 0.0f;
        s.pContactPeakFraction[slot] = 0.0f;
        clearColdWallParticleState(s, slot);
        clearColdWall2DParticleState(s, slot);
#endif
        s.pRng[slot] = rng;
        s.pOrigId[slot] =
            (static_cast<unsigned long long>(j) << 40)
          ^ static_cast<unsigned long long>(slot);
        consumed += injectionParcelMass;
        ++created;
    }

    s.sourceResidualMass[j] = available - consumed;
#if defined(UGKP_DEVELOPMENT_PROBES) && !GPU_OPERATOR_THERMAL
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = created;
    }
#endif
}
