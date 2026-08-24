#ifndef GPU_THERMAL_GPU_COLD_WALL_2D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_2D_DEVICE_CUH

__device__ inline double coldWall2DGroupSum
(
    double value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

__device__ inline double coldWall2DGroupMin
(
    double value,
    const unsigned int mask
)
{
    double other = __shfl_down_sync(mask, value, 4, 8);
    value = other < value ? other : value;
    other = __shfl_down_sync(mask, value, 2, 8);
    value = other < value ? other : value;
    other = __shfl_down_sync(mask, value, 1, 8);
    value = other < value ? other : value;
    return __shfl_sync(mask, value, 0, 8);
}

__device__ void clearColdWall2DParticleState
(
    DeviceState& s,
    const int particleI
)
{
    if (s.coldWall2DEnabled == 0)
    {
        return;
    }
    const int nodeBase =
        particleI*Foam::gpuThermal::coldWall2DNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWall2DRadialNodeCount;
    for (int node = 0; node < Foam::gpuThermal::coldWall2DNodeCount; ++node)
    {
        s.pCold2DNodeSpecificEnthalpy[nodeBase + node] = 0.0f;
    }
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        s.pCold2DRingContactAge[ringBase + ring] = 0.0f;
    }
    s.pCold2DFrozenArea[particleI] = 0.0f;
}

__device__ void initialiseColdWall2DParticleState
(
    DeviceState& s,
    const int particleI,
    const double temperatureK
)
{
    if
    (
        s.coldWall2DEnabled == 0
     || s.pCold2DNodeSpecificEnthalpy == nullptr
     || s.pCold2DRingContactAge == nullptr
     || s.pCold2DFrozenArea == nullptr
    )
    {
        asm("trap;");
    }
    const double initialEnthalpy =
        Foam::gpuThermal::coldWallSpecificEnthalpyJkg
        (
            temperatureK,
            s.coldWallSolidificationParameters
        );
    const int nodeBase =
        particleI*Foam::gpuThermal::coldWall2DNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWall2DRadialNodeCount;
    for (int node = 0; node < Foam::gpuThermal::coldWall2DNodeCount; ++node)
    {
        s.pCold2DNodeSpecificEnthalpy[nodeBase + node] =
            static_cast<float>(initialEnthalpy);
    }
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        s.pCold2DRingContactAge[ringBase + ring] = 0.0f;
    }
    s.pCold2DFrozenArea[particleI] = 0.0f;
}

__device__ bool advanceColdWall2DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int radialNode,
    const unsigned int groupMask,
    const double physicalVolumeM3,
    const double physicalMassKg,
    const double maximumAreaM2,
    const double intrinsicContactAreaM2,
    const double deltaTSeconds,
    const double wallTemperatureK,
    const double wallEffusivity,
    const double thermalAreaFactor,
    const double gasTemperatureK,
    const double gasConductanceWK,
    double& meanTemperatureK,
    double& surfaceTemperatureK,
    double& frozenFootprintAreaM2,
    double& wallEnergyJ
)
{
    const Foam::gpuThermal::ColdWall2DGeometry geometry =
        Foam::gpuThermal::coldWall2DGeometry
        (
            physicalVolumeM3,
            physicalMassKg,
            maximumAreaM2
        );
    if
    (
        !geometry.valid
     || !(intrinsicContactAreaM2 > 0.0)
     || intrinsicContactAreaM2 > maximumAreaM2
     || !(deltaTSeconds >= 0.0)
     || !(wallTemperatureK > 0.0)
     || !(wallEffusivity > 0.0)
     || !(thermalAreaFactor > 0.0)
     || thermalAreaFactor > 1.0
     || !(gasTemperatureK > 0.0)
     || gasConductanceWK < 0.0
    )
    {
        return false;
    }

    const int nodeBase =
        particleI*Foam::gpuThermal::coldWall2DNodeCount
      + radialNode*Foam::gpuThermal::coldWall2DAxialNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWall2DRadialNodeCount;
    double enthalpy[Foam::gpuThermal::coldWall2DAxialNodeCount];
    int localStateValid = 1;
    for
    (
        int axialNode = 0;
        axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
        ++axialNode
    )
    {
        enthalpy[axialNode] = static_cast<double>
        (
            s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode]
        );
        localStateValid = localStateValid
         && Foam::gpuThermal::finiteColdWallValue(enthalpy[axialNode])
         && enthalpy[axialNode] >= 0.0;
    }
    if (!__all_sync(groupMask, localStateValid))
    {
        return false;
    }
    double ringContactAgeS = static_cast<double>
    (
        s.pCold2DRingContactAge[ringBase + radialNode]
    );
    localStateValid =
        Foam::gpuThermal::finiteColdWallValue(ringContactAgeS)
     && ringContactAgeS >= 0.0;
    if (!__all_sync(groupMask, localStateValid))
    {
        return false;
    }
    const double wetAreaM2 = Foam::gpuThermal::coldWall2DRingWetAreaM2
    (
        geometry,
        intrinsicContactAreaM2,
        radialNode
    );
    double remainingTime = deltaTSeconds;
    double elapsedTime = 0.0;
    double localWallEnergyJ = 0.0;
    int substep = 0;
    while (remainingTime > 0.0 && substep < 256)
    {
        double temperature[Foam::gpuThermal::coldWall2DAxialNodeCount];
        double conductivity[Foam::gpuThermal::coldWall2DAxialNodeCount];
        double apparentCp[Foam::gpuThermal::coldWall2DAxialNodeCount];
        localStateValid = 1;
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            temperature[axialNode] =
                Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
                (
                    enthalpy[axialNode],
                    s.coldWallSolidificationParameters
                );
            conductivity[axialNode] =
                Foam::gpuThermal::coldWallThermalConductivity
                (
                    enthalpy[axialNode],
                    s.coldWallSolidificationParameters
                );
            apparentCp[axialNode] =
                Foam::gpuThermal::coldWallApparentSpecificHeat
                (
                    temperature[axialNode],
                    s.coldWallSolidificationParameters
                );
            localStateValid = localStateValid
             && Foam::gpuThermal::finiteColdWallValue
                (
                    temperature[axialNode]
                )
             && Foam::gpuThermal::finiteColdWallValue
                (
                    conductivity[axialNode]
                )
             && Foam::gpuThermal::finiteColdWallValue
                (
                    apparentCp[axialNode]
                )
             && temperature[axialNode] > 0.0
             && conductivity[axialNode] > 0.0
             && apparentCp[axialNode] > 0.0;
        }
        if (!__all_sync(groupMask, localStateValid))
        {
            return false;
        }

        const double localAge = ringContactAgeS + elapsedTime;
        const double particleWallResistance =
            0.5*geometry.axialNodeThicknessM/conductivity[0];
        const double transientWallResistance =
            s.coldWallSolidificationParameters.wallTransientResistance != 0
          ? ::sqrt(Foam::gpuThermal::coldWallPi*localAge)/wallEffusivity
          : 0.0;
        const double wallConductanceWK = wetAreaM2 > 0.0
          ? thermalAreaFactor*wetAreaM2
           /(
                particleWallResistance
              + s.coldWallSolidificationParameters.interfaceResistanceM2KW
              + transientWallResistance
              + DBL_MIN
            )
          : 0.0;
        const double localGasConductanceWK =
            gasConductanceWK
           /static_cast<double>(Foam::gpuThermal::coldWall2DRadialNodeCount);

        double minimumStableTime = DBL_MAX;
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            double conductanceSum = 0.0;
            if (axialNode > 0)
            {
                const double harmonic =
                    2.0*conductivity[axialNode]*conductivity[axialNode - 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode - 1]
                      + DBL_MIN
                    );
                conductanceSum +=
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
            }
            if (axialNode + 1 < Foam::gpuThermal::coldWall2DAxialNodeCount)
            {
                const double harmonic =
                    2.0*conductivity[axialNode]*conductivity[axialNode + 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode + 1]
                      + DBL_MIN
                    );
                conductanceSum +=
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
            }
            const double innerNeighborConductivity = __shfl_up_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            const double outerNeighborConductivity = __shfl_down_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            if (radialNode > 0)
            {
                conductanceSum +=
                    Foam::gpuThermal::coldWall2DRadialFaceConductanceWK
                    (
                        geometry,
                        radialNode - 1,
                        innerNeighborConductivity,
                        conductivity[axialNode]
                    );
            }
            if
            (
                radialNode + 1
              < Foam::gpuThermal::coldWall2DRadialNodeCount
            )
            {
                conductanceSum +=
                    Foam::gpuThermal::coldWall2DRadialFaceConductanceWK
                    (
                        geometry,
                        radialNode,
                        conductivity[axialNode],
                        outerNeighborConductivity
                    );
            }
            if (axialNode == 0)
            {
                conductanceSum += wallConductanceWK;
            }
            if
            (
                axialNode + 1
             == Foam::gpuThermal::coldWall2DAxialNodeCount
            )
            {
                conductanceSum += localGasConductanceWK;
            }
            if (conductanceSum > 0.0)
            {
                const double stableTime =
                    0.45*geometry.nodeMassKg*apparentCp[axialNode]
                   /conductanceSum;
                minimumStableTime = stableTime < minimumStableTime
                  ? stableTime : minimumStableTime;
            }
        }
        minimumStableTime = coldWall2DGroupMin
        (
            minimumStableTime,
            groupMask
        );
        const double stepTime = minimumStableTime < remainingTime
          ? minimumStableTime : remainingTime;
        if
        (
            !(stepTime > 0.0)
         || !Foam::gpuThermal::finiteColdWallValue(stepTime)
        )
        {
            return false;
        }

        double power[Foam::gpuThermal::coldWall2DAxialNodeCount];
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            double nodePower = 0.0;
            if (axialNode > 0)
            {
                const double harmonic =
                    2.0*conductivity[axialNode]*conductivity[axialNode - 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode - 1]
                      + DBL_MIN
                    );
                const double conductance =
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
                nodePower += conductance
                   *(temperature[axialNode - 1] - temperature[axialNode]);
            }
            if (axialNode + 1 < Foam::gpuThermal::coldWall2DAxialNodeCount)
            {
                const double harmonic =
                    2.0*conductivity[axialNode]*conductivity[axialNode + 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode + 1]
                      + DBL_MIN
                    );
                const double conductance =
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
                nodePower += conductance
                   *(temperature[axialNode + 1] - temperature[axialNode]);
            }
            const double innerNeighborTemperature = __shfl_up_sync
            (
                groupMask,
                temperature[axialNode],
                1,
                8
            );
            const double innerNeighborConductivity = __shfl_up_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            const double outerNeighborTemperature = __shfl_down_sync
            (
                groupMask,
                temperature[axialNode],
                1,
                8
            );
            const double outerNeighborConductivity = __shfl_down_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            if (radialNode > 0)
            {
                const double conductance =
                    Foam::gpuThermal::coldWall2DRadialFaceConductanceWK
                    (
                        geometry,
                        radialNode - 1,
                        innerNeighborConductivity,
                        conductivity[axialNode]
                    );
                nodePower += conductance
                   *(innerNeighborTemperature - temperature[axialNode]);
            }
            if
            (
                radialNode + 1
              < Foam::gpuThermal::coldWall2DRadialNodeCount
            )
            {
                const double conductance =
                    Foam::gpuThermal::coldWall2DRadialFaceConductanceWK
                    (
                        geometry,
                        radialNode,
                        conductivity[axialNode],
                        outerNeighborConductivity
                    );
                nodePower += conductance
                   *(outerNeighborTemperature - temperature[axialNode]);
            }
            if (axialNode == 0)
            {
                const double wallPower = wallConductanceWK
                   *(temperature[0] - wallTemperatureK);
                nodePower -= wallPower;
                localWallEnergyJ += wallPower*stepTime;
            }
            if
            (
                axialNode + 1
             == Foam::gpuThermal::coldWall2DAxialNodeCount
            )
            {
                nodePower += localGasConductanceWK
                   *(gasTemperatureK - temperature[axialNode]);
            }
            power[axialNode] = nodePower;
        }
        localStateValid = 1;
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            enthalpy[axialNode] +=
                stepTime*power[axialNode]/geometry.nodeMassKg;
            localStateValid = localStateValid
             && Foam::gpuThermal::finiteColdWallValue(enthalpy[axialNode])
             && enthalpy[axialNode] >= 0.0;
        }
        if (!__all_sync(groupMask, localStateValid))
        {
            return false;
        }
        elapsedTime += stepTime;
        remainingTime -= stepTime;
        if (remainingTime < 1.0e-12*deltaTSeconds)
        {
            remainingTime = 0.0;
        }
        ++substep;
    }
    if (remainingTime > 0.0)
    {
        return false;
    }

    double localEnthalpySum = 0.0;
    for
    (
        int axialNode = 0;
        axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
        ++axialNode
    )
    {
        s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode] =
            static_cast<float>(enthalpy[axialNode]);
        localEnthalpySum += enthalpy[axialNode];
    }
    if (wetAreaM2 > 0.0)
    {
        ringContactAgeS += deltaTSeconds;
    }
    s.pCold2DRingContactAge[ringBase + radialNode] =
        static_cast<float>(ringContactAgeS);

    const double connectedThickness =
        Foam::gpuThermal::coldWall2DConnectedSolidThicknessFraction
        (
            enthalpy,
            s.coldWallSolidificationParameters
        );
    const int pinned =
        wetAreaM2 > 0.0
     && connectedThickness
      >= s.coldWallSolidificationParameters.pinningThicknessFraction;
    double candidateFrozenArea = 0.0;
    int contiguousPinned = 1;
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        const int ringPinned = __shfl_sync(groupMask, pinned, ring, 8);
        const double ringWetArea = __shfl_sync
        (
            groupMask,
            wetAreaM2,
            ring,
            8
        );
        if (radialNode == 0 && contiguousPinned != 0)
        {
            if (ringPinned == 0 || !(ringWetArea > 0.0))
            {
                contiguousPinned = 0;
            }
            else
            {
                candidateFrozenArea += ringWetArea;
            }
        }
    }
    if (radialNode == 0)
    {
        const double oldFrozenArea = static_cast<double>
        (
            s.pCold2DFrozenArea[particleI]
        );
        frozenFootprintAreaM2 = candidateFrozenArea > oldFrozenArea
          ? candidateFrozenArea : oldFrozenArea;
        s.pCold2DFrozenArea[particleI] =
            static_cast<float>(frozenFootprintAreaM2);
    }
    frozenFootprintAreaM2 = __shfl_sync
    (
        groupMask,
        frozenFootprintAreaM2,
        0,
        8
    );
    const double enthalpySum = coldWall2DGroupSum
    (
        localEnthalpySum,
        groupMask
    );
    const double surfaceEnthalpySum = coldWall2DGroupSum
    (
        enthalpy[Foam::gpuThermal::coldWall2DAxialNodeCount - 1],
        groupMask
    );
    meanTemperatureK =
        Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (
            enthalpySum/static_cast<double>(Foam::gpuThermal::coldWall2DNodeCount),
            s.coldWallSolidificationParameters
        );
    surfaceTemperatureK =
        Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (
            surfaceEnthalpySum
           /static_cast<double>(Foam::gpuThermal::coldWall2DRadialNodeCount),
            s.coldWallSolidificationParameters
        );
    wallEnergyJ = coldWall2DGroupSum(localWallEnergyJ, groupMask);
    return
        Foam::gpuThermal::finiteColdWallValue(meanTemperatureK)
     && Foam::gpuThermal::finiteColdWallValue(surfaceTemperatureK)
     && Foam::gpuThermal::finiteColdWallValue(frozenFootprintAreaM2)
     && Foam::gpuThermal::finiteColdWallValue(wallEnergyJ)
     && meanTemperatureK > 0.0
     && surfaceTemperatureK > 0.0
     && frozenFootprintAreaM2 >= 0.0
     && frozenFootprintAreaM2 <= maximumAreaM2;
}

__global__ void relaxColdWall2DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    if
    (
        s.coldWall2DEnabled == 0
     || blockDim.x < Foam::gpuThermal::coldWall2DRadialNodeCount
     || blockDim.x%Foam::gpuThermal::coldWall2DRadialNodeCount != 0
     || blockDim.x%32 != 0
    )
    {
        return;
    }
    const int radialNode = threadIdx.x
       %Foam::gpuThermal::coldWall2DRadialNodeCount;
    const int warpLane = threadIdx.x & 31;
    const unsigned int groupMask = 0xffu << (warpLane & ~7);
    const int groupsPerBlock = blockDim.x
       /Foam::gpuThermal::coldWall2DRadialNodeCount;
    const int group = blockIdx.x*groupsPerBlock
      + threadIdx.x/Foam::gpuThermal::coldWall2DRadialNodeCount;
    const int groupStride = gridDim.x*groupsPerBlock;
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for (int entry = group; entry < nWallBound; entry += groupStride)
    {
        int particleI = -1;
        int eligible = 0;
        int wallFaceI = -1;
        int wallState = Foam::gpuThermal::particleWallMobile;
        if (radialNode == 0)
        {
            particleI = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
            if (particleI < 0 || particleI >= s.particleCapacity)
            {
                asm("trap;");
            }
            wallFaceI = s.pStuckFaceId[particleI];
            wallState = static_cast<int>(s.pStuck[particleI]);
            eligible =
                s.pStatus[particleI] != 0
             && wallState != Foam::gpuThermal::particleWallMobile
             && wallFaceI >= 0
             && wallFaceI < s.nFaces
             && s.particleStuckCandidateMask[wallFaceI]
                == Foam::gpuThermal::particleWallColdWall2D;
        }
        particleI = __shfl_sync(groupMask, particleI, 0, 8);
        eligible = __shfl_sync(groupMask, eligible, 0, 8);
        wallFaceI = __shfl_sync(groupMask, wallFaceI, 0, 8);
        wallState = __shfl_sync(groupMask, wallState, 0, 8);
        if (eligible == 0)
        {
            continue;
        }

        int cellI = 0;
        double particleDiameterM = 0.0;
        double particleTemperatureK = 0.0;
        double gasTemperatureK = 0.0;
        double gasConductanceWK = 0.0;
        double physicalVolumeM3 = 0.0;
        double physicalMassKg = 0.0;
        double parcelMultiplicity = 0.0;
        double wallEffusivity = 0.0;
        double maximumAreaM2 = 0.0;
        double durationS = 0.0;
        double peakTimeFraction = 0.0;
        double damageAreaM2 = 0.0;
        double age0S = 0.0;
        double age1S = 0.0;
        double thermalDeltaTS = 0.0;
        double intrinsicAreaM2 = 0.0;
        double thermalAreaFactor = 0.0;
        int finiteContact = 0;
        if (radialNode == 0)
        {
            cellI = s.pCellId[particleI];
            if (cellI < 0 || cellI >= s.nCells)
            {
                asm("trap;");
            }
            finiteContact =
                wallState == Foam::gpuThermal::particleWallTransientRebound
             || wallState == Foam::gpuThermal::particleWallTransientDeposit;
            s.pux[particleI] = 0.0;
            s.puy[particleI] = 0.0;
            s.puz[particleI] = 0.0;
            if (!finiteContact)
            {
                s.puxOld[particleI] = 0.0;
                s.puyOld[particleI] = 0.0;
                s.puzOld[particleI] = 0.0;
                const double alphaTheta = clampRange
                (
                    finiteOr(s.thetaDragAlpha[cellI], 1.0),
                    0.0,
                    1.0
                );
                s.pTheta[particleI] =
                    clampMin(finiteOr(s.pTheta[particleI], 0.0), 0.0)
                   *alphaTheta;
            }
            particleDiameterM = clampMin
            (
                finiteOr(s.pd[particleI], s.particleDiameterFallback),
                1.0e-12
            );
            particleTemperatureK = clampRange
            (
                finiteOr(s.pT[particleI], s.TpMin),
                s.TpMin,
                s.TpMax
            );
            gasTemperatureK = clampRange
            (
                finiteOr(s.couplingTgasOld[cellI], s.TgasMin),
                s.TgasMin,
                1.0e30
            );
            physicalVolumeM3 =
                (Foam::gpuThermal::finiteContactPi/6.0)
               *particleDiameterM*particleDiameterM*particleDiameterM;
            const Foam::gpuThermal::AluminaLiquidProperties material =
                Foam::gpuThermal::liquidAluminaProperties
                (
                    particleTemperatureK
                );
            physicalMassKg = material.densityKgM3*physicalVolumeM3;
            parcelMultiplicity = s.pm[particleI]/physicalMassKg;
            if
            (
                s.solveParticleTemperature != 0
             && s.particleGasHeatTransferModelId != 0
            )
            {
                const double rhoG = clampMin
                (
                    finiteOr(s.couplingRhoOld[cellI], s.rhoMin),
                    s.rhoMin
                );
                const double ugx = finiteOr(s.couplingUxOld[cellI], 0.0);
                const double ugy = finiteOr(s.couplingUyOld[cellI], 0.0);
                const double ugz = finiteOr(s.couplingUzOld[cellI], 0.0);
                const double re =
                    rhoG*particleDiameterM*sqrt(sqr3(ugx, ugy, ugz))
                   /clampMin(s.gasMu, 1.0e-30);
                const double nusselt =
                    2.0 + 0.6*sqrt(clampMin(re, 0.0))*s.gasPrOneThird;
                const double particleCp =
                    particleSpecificHeatDevice(particleTemperatureK);
                const double rate =
                    6.0*nusselt*molecularGasConductivity(s)
                   /(
                        s.rhoSolid*particleCp
                       *particleDiameterM*particleDiameterM
                      + 1.0e-300
                    );
                gasConductanceWK = rate*physicalMassKg*particleCp;
            }
            wallEffusivity = s.particleWallEffusivityByFace != nullptr
              ? s.particleWallEffusivityByFace[wallFaceI]
              : sqrt
                (
                    s.particleWallDensityKgM3
                   *s.particleWallSpecificHeatJkgK
                   *s.particleWallConductivityWmK
                );
            maximumAreaM2 = static_cast<double>
            (
                s.pContactMaximumArea[particleI]
            );
            durationS = static_cast<double>(s.pContactDuration[particleI]);
            peakTimeFraction = static_cast<double>
            (
                s.pContactPeakFraction[particleI]
            );
            damageAreaM2 = static_cast<double>(s.pDepositionArea[particleI]);
            const double contactAreaScale =
                s.particleWallContactAreaScale[wallFaceI];
            if
            (
                !(physicalMassKg > 0.0)
             || !(parcelMultiplicity > 0.0)
             || !(wallEffusivity > 0.0)
             || !(maximumAreaM2 > 0.0)
             || !(durationS > 0.0)
             || !(peakTimeFraction > 0.0)
             || !(peakTimeFraction < 1.0)
             || damageAreaM2 < 0.0
             || !(contactAreaScale > 0.0)
             || contactAreaScale > 1.0
            )
            {
                asm("trap;");
            }
            if (finiteContact)
            {
                age0S = clampRange
                (
                    finiteOr(s.pTheta[particleI], 0.0),
                    0.0,
                    durationS
                );
                thermalDeltaTS = clampRange(dt, 0.0, durationS - age0S);
                age1S = age0S + thermalDeltaTS;
                const double ageMidS = age0S + 0.5*thermalDeltaTS;
                const double kinematicAreaMidM2 = maximumAreaM2
                   *Foam::gpuThermal::normalizedKinematicArea
                    (
                        ageMidS/durationS,
                        peakTimeFraction
                    );
                intrinsicAreaM2 = clampMin
                (
                    fmax
                    (
                        kinematicAreaMidM2,
                        static_cast<double>(s.pCold2DFrozenArea[particleI])
                    ) - damageAreaM2,
                    0.0
                );
                thermalAreaFactor =
                    s.particleWallReflectionHeatTransferEfficiency
                   *contactAreaScale;
            }
            else
            {
                thermalDeltaTS = dt;
                intrinsicAreaM2 = damageAreaM2;
                thermalAreaFactor =
                    s.particleWallDepositionHeatTransferEfficiency
                   *contactAreaScale;
            }
        }

#define COLD_WALL_2D_BROADCAST(V) \
        V = __shfl_sync(groupMask, V, 0, 8)
        COLD_WALL_2D_BROADCAST(cellI);
        COLD_WALL_2D_BROADCAST(particleDiameterM);
        COLD_WALL_2D_BROADCAST(particleTemperatureK);
        COLD_WALL_2D_BROADCAST(gasTemperatureK);
        COLD_WALL_2D_BROADCAST(gasConductanceWK);
        COLD_WALL_2D_BROADCAST(physicalVolumeM3);
        COLD_WALL_2D_BROADCAST(physicalMassKg);
        COLD_WALL_2D_BROADCAST(parcelMultiplicity);
        COLD_WALL_2D_BROADCAST(wallEffusivity);
        COLD_WALL_2D_BROADCAST(maximumAreaM2);
        COLD_WALL_2D_BROADCAST(durationS);
        COLD_WALL_2D_BROADCAST(peakTimeFraction);
        COLD_WALL_2D_BROADCAST(damageAreaM2);
        COLD_WALL_2D_BROADCAST(age0S);
        COLD_WALL_2D_BROADCAST(age1S);
        COLD_WALL_2D_BROADCAST(thermalDeltaTS);
        COLD_WALL_2D_BROADCAST(intrinsicAreaM2);
        COLD_WALL_2D_BROADCAST(thermalAreaFactor);
        COLD_WALL_2D_BROADCAST(finiteContact);
#undef COLD_WALL_2D_BROADCAST

        double meanTemperatureK = particleTemperatureK;
        double surfaceTemperatureK = particleTemperatureK;
        double frozenFootprintAreaM2 = static_cast<double>
        (
            s.pCold2DFrozenArea[particleI]
        );
        double wallEnergyJ = 0.0;
        if (thermalDeltaTS > 0.0 && intrinsicAreaM2 > 0.0)
        {
            const bool thermalValid = advanceColdWall2DThermalGroup
            (
                s,
                particleI,
                radialNode,
                groupMask,
                physicalVolumeM3,
                physicalMassKg,
                maximumAreaM2,
                intrinsicAreaM2,
                thermalDeltaTS,
                s.gasBoundaryT[wallFaceI],
                wallEffusivity,
                thermalAreaFactor,
                gasTemperatureK,
                gasConductanceWK,
                meanTemperatureK,
                surfaceTemperatureK,
                frozenFootprintAreaM2,
                wallEnergyJ
            );
            if (!thermalValid)
            {
                asm("trap;");
            }
            if (radialNode == 0)
            {
                s.pT[particleI] = meanTemperatureK;
                atomicAddParticleWallEnergyByFace
                (
                    s,
                    finiteContact
                  ? s.particleWallReflectedEnergy
                  : s.particleWallDepositedEnergy,
                    wallFaceI,
                    parcelMultiplicity*wallEnergyJ
                );
            }
        }

        int transition = 0;
        double longDepositAreaM2 = 0.0;
        if (radialNode == 0 && finiteContact)
        {
            s.pTheta[particleI] = age1S;
            const double theta1 = age1S/durationS;
            const double kinematicAreaM2 = maximumAreaM2
               *Foam::gpuThermal::normalizedKinematicArea
                (
                    theta1,
                    peakTimeFraction
                );
            const double effectiveContactAreaM2 =
                fmax(kinematicAreaM2, frozenFootprintAreaM2) - damageAreaM2;
            bool detach = false;
            bool enterLongDeposit = false;
            if
            (
                wallState
             == Foam::gpuThermal::particleWallTransientRebound
            )
            {
                detach = !(effectiveContactAreaM2 > 0.0);
                enterLongDeposit =
                    !detach
                 && !(kinematicAreaM2 > 0.0)
                 && frozenFootprintAreaM2 > 0.0;
                longDepositAreaM2 = effectiveContactAreaM2;
            }
            else
            {
                const Foam::gpuThermal::CapillaryDetachmentState capillary =
                    Foam::gpuThermal::evaluateCapillaryDetachmentState
                    (
                        s.pT[particleI],
                        s.pd[particleI],
                        s.particleWallAdhesionEnergyScale,
                        s.particleWallContactAngleCosine
                    );
                if (!capillary.valid)
                {
                    asm("trap;");
                }
                const double targetContactAreaM2 = fmin
                (
                    capillary.equilibriumContactAreaM2,
                    maximumAreaM2
                );
                longDepositAreaM2 =
                    fmax(targetContactAreaM2, frozenFootprintAreaM2)
                   -damageAreaM2;
                detach = !(effectiveContactAreaM2 > 0.0);
                enterLongDeposit =
                    !detach
                 && theta1 >= peakTimeFraction
                 &&
                    (
                        kinematicAreaM2 <= targetContactAreaM2
                     || !(kinematicAreaM2 > 0.0)
                    );
            }
            transition = detach ? 1 : (enterLongDeposit ? 2 : 0);
            if (transition == 1)
            {
                s.pux[particleI] = s.puxOld[particleI];
                s.puy[particleI] = s.puyOld[particleI];
                s.puz[particleI] = s.puzOld[particleI];
                s.pStuck[particleI] = Foam::gpuThermal::particleWallMobile;
                s.pStuckFaceId[particleI] = -1;
                s.pTheta[particleI] = 0.0;
                s.pDepositionArea[particleI] = 0.0f;
                s.pContactDuration[particleI] = 0.0f;
                s.pContactMaximumArea[particleI] = 0.0f;
                s.pContactPeakFraction[particleI] = 0.0f;
            }
            else if (transition == 2)
            {
                if
                (
                    !(longDepositAreaM2 > 0.0)
                 || longDepositAreaM2 > static_cast<double>(FLT_MAX)
                )
                {
                    asm("trap;");
                }
                s.pStuck[particleI] = Foam::gpuThermal::particleWallDeposited;
                s.pTheta[particleI] = 0.0;
                s.pDepositionArea[particleI] =
                    static_cast<float>(longDepositAreaM2);
                s.puxOld[particleI] = 0.0;
                s.puyOld[particleI] = 0.0;
                s.puzOld[particleI] = 0.0;
            }
        }
        transition = __shfl_sync(groupMask, transition, 0, 8);
        if (transition == 1)
        {
            const int nodeBase =
                particleI*Foam::gpuThermal::coldWall2DNodeCount
              + radialNode*Foam::gpuThermal::coldWall2DAxialNodeCount;
            for
            (
                int axialNode = 0;
                axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
                ++axialNode
            )
            {
                s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode] = 0.0f;
            }
            s.pCold2DRingContactAge
            [
                particleI*Foam::gpuThermal::coldWall2DRadialNodeCount
              + radialNode
            ] = 0.0f;
            if (radialNode == 0)
            {
                s.pCold2DFrozenArea[particleI] = 0.0f;
            }
        }
    }
}

#endif
