#include "GpuPrecisionTypes.H"
#include <type_traits>
#include "GpuColdWallThermalCarry.cuh"
#ifndef GPU_THERMAL_GPU_COLD_WALL_2D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_2D_DEVICE_CUH

template<class Scalar>
__device__ inline Scalar coldWall2DGroupSum
(
    Scalar value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

__device__ inline GpuTime coldWall2DGroupMin
(
    GpuTime value,
    const unsigned int mask
)
{
    GpuTime other = __shfl_down_sync(mask, value, 4, 8);
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
        s.pCold2DNodeSpecificEnthalpy[nodeBase + node] = 0.0;
    }
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        s.pCold2DRingContactAge[ringBase + ring] = 0.0;
    }
    s.pCold2DFrozenArea[particleI] = 0.0;
}

__device__ void initialiseColdWall2DParticleState
(
    DeviceState& s,
    const int particleI,
    const GpuReal temperatureK
)
{
    if
    (
        s.pCold2DNodeSpecificEnthalpy == nullptr
     || s.pCold2DRingContactAge == nullptr
     || s.pCold2DFrozenArea == nullptr
    )
    {
        asm("trap;");
    }
    const GpuReal initialEnthalpy =
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
            static_cast<typename std::remove_reference<decltype(s.pCold2DNodeSpecificEnthalpy[0])>::type>(initialEnthalpy);
    }
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        s.pCold2DRingContactAge[ringBase + ring] = 0.0;
    }
#ifdef GPU_PAIRED_PARTICLE_HEAT


    s.pThermalEnthalpyResidual[particleI] += double(initialEnthalpy)
      - double(s.pCold2DNodeSpecificEnthalpy[nodeBase]);
#endif
    s.pCold2DFrozenArea[particleI] = 0.0;
}

__device__ bool advanceColdWall2DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int radialNode,
    const unsigned int groupMask,
    const GpuReal physicalVolumeM3,
    const GpuReal physicalMassKg,
    const GpuReal maximumAreaM2,
    const GpuReal intrinsicContactAreaM2,
    const GpuTime deltaTSeconds,
    const GpuReal wallTemperatureK,
    const GpuReal wallEffusivity,
    const GpuReal thermalAreaFactor,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK,
    GpuReal& meanTemperatureK,
    GpuReal& surfaceTemperatureK,
    GpuReal& frozenFootprintAreaM2,
    GpuReal& wallEnergyJ
)
{
    const Foam::gpuThermal::ColdWall2DGeometry geometry =
        Foam::gpuThermal::coldWall2DGeometry
        (
            physicalVolumeM3,
            physicalMassKg,
            maximumAreaM2
        );

    const bool backingValid = s.pCold2DNodeSpecificEnthalpy != nullptr
     && s.pCold2DRingContactAge != nullptr
     && s.pCold2DFrozenArea != nullptr
#ifdef GPU_PAIRED_PARTICLE_HEAT
     && s.pThermalEnthalpyResidual != nullptr
#endif
     ;
    if (!__all_sync(groupMask, backingValid)) return false;

    const int nodeBase =
        particleI*Foam::gpuThermal::coldWall2DNodeCount
      + radialNode*Foam::gpuThermal::coldWall2DAxialNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWall2DRadialNodeCount;
#ifdef GPU_PAIRED_PARTICLE_HEAT
    const double oldCarry=s.pThermalEnthalpyResidual[particleI];
    double localOldMean=0.0;
#endif
    ColdWallThermalReal enthalpy[Foam::gpuThermal::coldWall2DAxialNodeCount];
    for
    (
        int axialNode = 0;
        axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
        ++axialNode
    )
    {
        enthalpy[axialNode] = static_cast<ColdWallThermalReal>
        (
            s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode]
        );
#ifdef GPU_PAIRED_PARTICLE_HEAT
        localOldMean += enthalpy[axialNode]/64.0;
        enthalpy[axialNode] += oldCarry;
#endif
    }

#ifdef GPU_PAIRED_PARTICLE_HEAT
    const double oldMean=coldWall2DGroupSum(localOldMean,groupMask)+oldCarry;
    if(deltaTSeconds==0.0) {
        meanTemperatureK=static_cast<GpuReal>(coldWallCarryTemperature(oldMean,s.coldWallSolidificationParameters));
        const double top=coldWall2DGroupSum(enthalpy[7]/8.0,groupMask);
        surfaceTemperatureK=static_cast<GpuReal>(coldWallCarryTemperature(top,s.coldWallSolidificationParameters));
        frozenFootprintAreaM2=s.pCold2DFrozenArea[particleI];wallEnergyJ=GPU_R(0.0);return true;
    }
#endif
    GpuTime ringContactAgeS = static_cast<GpuTime>
    (
        s.pCold2DRingContactAge[ringBase + radialNode]
    );

    const ColdWallThermalReal wetAreaM2 = Foam::gpuThermal::coldWall2DRingWetAreaM2
    (
        geometry,
        intrinsicContactAreaM2,
        radialNode
    );
    const GpuTime timeDirection = deltaTSeconds < 0.0 ? -1.0 : 1.0;
    GpuTime remainingTime = ::fabs(deltaTSeconds);
    GpuTime elapsedTime = GPU_R(0.0);
    ColdWallThermalReal localWallEnergyJ = GPU_R(0.0);
#ifdef GPU_PAIRED_PARTICLE_HEAT
    double localGasEnergyJ=0.0;
#endif


    int substep = deltaTSeconds < 0.0 ? 256 : 0;
    GpuTime remainingCorrection = 0.0;

    const bool nonfiniteInterval = !Foam::gpuThermal::finiteColdWallValue(deltaTSeconds);
    while (remainingTime > GPU_R(0.0) || remainingCorrection < 0.0 || nonfiniteInterval)
    {
        ColdWallThermalReal temperature[Foam::gpuThermal::coldWall2DAxialNodeCount];
        ColdWallThermalReal conductivity[Foam::gpuThermal::coldWall2DAxialNodeCount];
        ColdWallThermalReal apparentCp[Foam::gpuThermal::coldWall2DAxialNodeCount];
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            temperature[axialNode] =
                coldWallCarryTemperature
                (
                    enthalpy[axialNode],
                    s.coldWallSolidificationParameters
                );
#ifdef GPU_PAIRED_PARTICLE_HEAT
            const double fraction=coldWallCarryFraction(temperature[axialNode],s.coldWallSolidificationParameters);
            conductivity[axialNode]=fraction*s.coldWallSolidificationParameters.solidThermalConductivityWmK
              +(1.0-fraction)*Foam::gpuThermal::aluminaThermalConductivityWmK;
#else
            conductivity[axialNode] =
                Foam::gpuThermal::coldWallThermalConductivity
                (
                    enthalpy[axialNode],
                    s.coldWallSolidificationParameters
                );
#endif
            apparentCp[axialNode] =
                coldWallCarrySpecificHeat
                (
                    temperature[axialNode],
                    s.coldWallSolidificationParameters
                );
        }

        const GpuTime localAge = ringContactAgeS + elapsedTime;
        const ColdWallThermalReal particleWallResistance =
            GPU_R(0.5)*geometry.axialNodeThicknessM/conductivity[0];
        const ColdWallThermalReal transientWallResistance =
            s.coldWallSolidificationParameters.wallTransientResistance != 0
          ? ::sqrt(Foam::gpuThermal::coldWallPi*localAge)/wallEffusivity
          : GPU_R(0.0);
        const ColdWallThermalReal wallConductanceWK = wetAreaM2 > GPU_R(0.0)
          ? thermalAreaFactor*wetAreaM2
           /(
                particleWallResistance
              + s.coldWallSolidificationParameters.interfaceResistanceM2KW
              + transientWallResistance
              + GPU_REAL_MIN
            )
          : GPU_R(0.0);
        const ColdWallThermalReal localGasConductanceWK =
            gasConductanceWK
           /static_cast<ColdWallThermalReal>(Foam::gpuThermal::coldWall2DRadialNodeCount);

        GpuTime minimumStableTime = DBL_MAX;
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            ColdWallThermalReal conductanceSum = GPU_R(0.0);
            if (axialNode > 0)
            {
                const ColdWallThermalReal harmonic =
                    GPU_R(2.0)*conductivity[axialNode]*conductivity[axialNode - 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode - 1]
                      + GPU_REAL_MIN
                    );
                conductanceSum +=
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
            }
            if (axialNode + 1 < Foam::gpuThermal::coldWall2DAxialNodeCount)
            {
                const ColdWallThermalReal harmonic =
                    GPU_R(2.0)*conductivity[axialNode]*conductivity[axialNode + 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode + 1]
                      + GPU_REAL_MIN
                    );
                conductanceSum +=
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
            }
            const ColdWallThermalReal innerNeighborConductivity = __shfl_up_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            const ColdWallThermalReal outerNeighborConductivity = __shfl_down_sync
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
            if (conductanceSum > GPU_R(0.0))
            {
                const GpuTime stableTime =
                    GPU_R(0.45)*geometry.nodeMassKg*apparentCp[axialNode]
                   /conductanceSum;
                const GpuTime stableMagnitude = ::fabs(stableTime);
                minimumStableTime = stableMagnitude < minimumStableTime
                  ? stableMagnitude : minimumStableTime;
            }
        }
        minimumStableTime = coldWall2DGroupMin
        (
            minimumStableTime,
            groupMask
        );
        const GpuTime availableTime = substep < 256
          ? remainingTime : remainingTime - remainingCorrection;
        const GpuTime stableMagnitude = minimumStableTime < availableTime
          ? minimumStableTime : availableTime;


        const GpuTime stepMagnitude = !nonfiniteInterval
          && stableMagnitude > GPU_R(0.0)
          && Foam::gpuThermal::finiteColdWallValue(stableMagnitude)
          ? stableMagnitude : availableTime;
        const GpuTime stepTime = timeDirection*stepMagnitude;

        ColdWallThermalReal power[Foam::gpuThermal::coldWall2DAxialNodeCount];
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            ColdWallThermalReal nodePower = GPU_R(0.0);
            if (axialNode > 0)
            {
                const ColdWallThermalReal harmonic =
                    GPU_R(2.0)*conductivity[axialNode]*conductivity[axialNode - 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode - 1]
                      + GPU_REAL_MIN
                    );
                const ColdWallThermalReal conductance =
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
                nodePower += conductance
                   *(temperature[axialNode - 1] - temperature[axialNode]);
            }
            if (axialNode + 1 < Foam::gpuThermal::coldWall2DAxialNodeCount)
            {
                const ColdWallThermalReal harmonic =
                    GPU_R(2.0)*conductivity[axialNode]*conductivity[axialNode + 1]
                   /(
                        conductivity[axialNode]
                      + conductivity[axialNode + 1]
                      + GPU_REAL_MIN
                    );
                const ColdWallThermalReal conductance =
                    harmonic*geometry.ringAreaM2
                   /geometry.axialNodeThicknessM;
                nodePower += conductance
                   *(temperature[axialNode + 1] - temperature[axialNode]);
            }
            const ColdWallThermalReal innerNeighborTemperature = __shfl_up_sync
            (
                groupMask,
                temperature[axialNode],
                1,
                8
            );
            const ColdWallThermalReal innerNeighborConductivity = __shfl_up_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            const ColdWallThermalReal outerNeighborTemperature = __shfl_down_sync
            (
                groupMask,
                temperature[axialNode],
                1,
                8
            );
            const ColdWallThermalReal outerNeighborConductivity = __shfl_down_sync
            (
                groupMask,
                conductivity[axialNode],
                1,
                8
            );
            if (radialNode > 0)
            {
                const ColdWallThermalReal conductance =
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
                const ColdWallThermalReal conductance =
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
                const ColdWallThermalReal wallPower = wallConductanceWK
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
                const ColdWallThermalReal gasPower=localGasConductanceWK
                   *(gasTemperatureK - temperature[axialNode]);
                nodePower += gasPower;
#ifdef GPU_PAIRED_PARTICLE_HEAT
                localGasEnergyJ += gasPower*stepTime;
#endif
            }
            power[axialNode] = nodePower;
        }
        for
        (
            int axialNode = 0;
            axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
            ++axialNode
        )
        {
            enthalpy[axialNode] +=
                stepTime*power[axialNode]/geometry.nodeMassKg;
        }

        if (nonfiniteInterval)
        {
            elapsedTime += stepTime;
            break;
        }
        if (substep < 256)
        {
            elapsedTime += stepTime;
            remainingTime -= stepTime;
            if (remainingTime < GPU_R(1.0e-12)*deltaTSeconds)
            {
                remainingTime = GPU_R(0.0);
            }
            ++substep;
        }
        else
        {
            const GpuTime correctedStep = -stepMagnitude - remainingCorrection;
            const GpuTime nextRemaining = remainingTime + correctedStep;
            remainingCorrection = (nextRemaining - remainingTime) - correctedStep;
            remainingTime = nextRemaining;
            elapsedTime = timeDirection*((::fabs(deltaTSeconds)-remainingTime)+remainingCorrection);
        }
    }

#ifdef GPU_PAIRED_PARTICLE_HEAT
    wallEnergyJ=static_cast<GpuReal>(coldWall2DGroupSum(localWallEnergyJ,groupMask));
    const GpuReal multiplicity=s.pm[particleI]/physicalMassKg;
    const GpuReal representedWallEnergy=multiplicity*wallEnergyJ;
    const double gasSpecificIncrement=coldWall2DGroupSum(localGasEnergyJ,groupMask)/double(physicalMassKg);
    const double meanIncrement=gasSpecificIncrement-double(representedWallEnergy)/double(s.pm[particleI]);
    double localCandidateMean=0.0;
    for(int node=0;node<8;++node)localCandidateMean+=enthalpy[node]/64.0;
    const double correction=(oldMean-coldWall2DGroupSum(localCandidateMean,groupMask))+meanIncrement;
    double localStorageChange=0.0;
    for(int node=0;node<8;++node) {
        enthalpy[node]+=correction;
        const float stored=ugkwp::coldWallStoredEnthalpyFloor(enthalpy[node]);
        localStorageChange+=(double(stored)-double(s.pCold2DNodeSpecificEnthalpy[nodeBase+node]))/64.0;
    }
    const double storageChange=coldWall2DGroupSum(localStorageChange,groupMask);
    const double nextCarry=(oldCarry-storageChange)+meanIncrement;
    const double actualSpecificChange=storageChange+(nextCarry-oldCarry);
    if(radialNode==0)s.pThermalEnthalpyResidual[particleI]=nextCarry;
    // Phase and output caches use exactly the published float profile + carry.
    for(int node=0;node<8;++node) {
        const float stored=ugkwp::coldWallStoredEnthalpyFloor(enthalpy[node]);
        s.pCold2DNodeSpecificEnthalpy[nodeBase+node]=stored;
        enthalpy[node]=double(stored)+nextCarry;
    }
#endif
    ColdWallThermalReal localEnthalpySum = GPU_R(0.0);
    for
    (
        int axialNode = 0;
        axialNode < Foam::gpuThermal::coldWall2DAxialNodeCount;
        ++axialNode
    )
    {
#ifndef GPU_PAIRED_PARTICLE_HEAT
        s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode] =
            enthalpy[axialNode];
#endif
        localEnthalpySum += enthalpy[axialNode];
    }
    if (wetAreaM2 > GPU_R(0.0))
    {
        ringContactAgeS += deltaTSeconds;
    }
    s.pCold2DRingContactAge[ringBase + radialNode] =
        static_cast<GpuTime>(ringContactAgeS);

#ifdef GPU_PAIRED_PARTICLE_HEAT
    double connectedFraction=1.0,connectedThickness=0.0;
    for(int node=0;node<8;++node) {
        const double f=coldWallCarryFraction(coldWallCarryTemperature(enthalpy[node],s.coldWallSolidificationParameters),s.coldWallSolidificationParameters);
        connectedFraction=f<connectedFraction?f:connectedFraction;
        connectedThickness+=connectedFraction/8.0;
    }
#else
    const ColdWallThermalReal connectedThickness =
        Foam::gpuThermal::coldWall2DConnectedSolidThicknessFraction
        (
            enthalpy,
            s.coldWallSolidificationParameters
        );
#endif
    const int pinned =
        wetAreaM2 > GPU_R(0.0)
     && connectedThickness
      >= s.coldWallSolidificationParameters.pinningThicknessFraction;
    ColdWallThermalReal candidateFrozenArea = GPU_R(0.0);
    int contiguousPinned = 1;
    for
    (
        int ring = 0;
        ring < Foam::gpuThermal::coldWall2DRadialNodeCount;
        ++ring
    )
    {
        const int ringPinned = __shfl_sync(groupMask, pinned, ring, 8);
        const ColdWallThermalReal ringWetArea = __shfl_sync
        (
            groupMask,
            wetAreaM2,
            ring,
            8
        );
        if (radialNode == 0 && contiguousPinned != 0)
        {
            if (ringPinned == 0 || !(ringWetArea > GPU_R(0.0)))
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
        const ColdWallThermalReal oldFrozenArea = static_cast<ColdWallThermalReal>
        (
            s.pCold2DFrozenArea[particleI]
        );
        frozenFootprintAreaM2 = candidateFrozenArea > oldFrozenArea
          ? candidateFrozenArea : oldFrozenArea;
        s.pCold2DFrozenArea[particleI] =
            frozenFootprintAreaM2;
    }
    frozenFootprintAreaM2 = __shfl_sync
    (
        groupMask,
        frozenFootprintAreaM2,
        0,
        8
    );
    const ColdWallThermalReal enthalpySum = coldWall2DGroupSum
    (
        localEnthalpySum,
        groupMask
    );
    const ColdWallThermalReal surfaceEnthalpySum = coldWall2DGroupSum
    (
        enthalpy[Foam::gpuThermal::coldWall2DAxialNodeCount - 1],
        groupMask
    );
    meanTemperatureK =
        coldWallCarryTemperature
        (
            enthalpySum/static_cast<ColdWallThermalReal>(Foam::gpuThermal::coldWall2DNodeCount),
            s.coldWallSolidificationParameters
        );
    surfaceTemperatureK =
        coldWallCarryTemperature
        (
            surfaceEnthalpySum
           /static_cast<ColdWallThermalReal>(Foam::gpuThermal::coldWall2DRadialNodeCount),
            s.coldWallSolidificationParameters
        );
    wallEnergyJ = coldWall2DGroupSum(localWallEnergyJ, groupMask);
#ifdef GPU_PAIRED_PARTICLE_HEAT
    if(radialNode==0 && gasConductanceWK!=GPU_R(0.0)) {
        const int cell=s.pCellId[particleI];
        // This is the actual accepted inventory delta. Wall heat is external
        // to the gas-particle pair, and is added back using its published value.
        atomicAdd(&s.particleHeatBudget[cell],double(s.pm[particleI])*actualSpecificChange
          +double(representedWallEnergy));
    }
#endif
    return true;
}

__global__ void relaxColdWall2DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const GpuTime dt
)
{
    DeviceState& s = *sp;
    if
    (
        s.coldWall2DEnabled == 0
     || blockDim.x < Foam::gpuThermal::coldWall2DRadialNodeCount
     || blockDim.x%Foam::gpuThermal::coldWall2DRadialNodeCount != 0
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
        GpuReal particleDiameterM = GPU_R(0.0);
        GpuReal particleTemperatureK = GPU_R(0.0);
        GpuReal gasTemperatureK = GPU_R(0.0);
        GpuReal gasConductanceWK = GPU_R(0.0);
        GpuReal physicalVolumeM3 = GPU_R(0.0);
        GpuReal physicalMassKg = GPU_R(0.0);
        GpuReal parcelMultiplicity = GPU_R(0.0);
        GpuReal wallEffusivity = GPU_R(0.0);
        GpuReal maximumAreaM2 = GPU_R(0.0);
        GpuTime durationS = GPU_R(0.0);
        GpuReal peakTimeFraction = GPU_R(0.0);
        GpuReal damageAreaM2 = GPU_R(0.0);
        GpuTime age0S = GPU_R(0.0);
        GpuTime age1S = GPU_R(0.0);
        GpuTime thermalDeltaTS = GPU_R(0.0);
        GpuReal intrinsicAreaM2 = GPU_R(0.0);
        GpuReal thermalAreaFactor = GPU_R(0.0);
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
            s.pux[particleI] = GPU_R(0.0);
            s.puy[particleI] = GPU_R(0.0);
            s.puz[particleI] = GPU_R(0.0);
            if (!finiteContact)
            {
                s.puxOld[particleI] = GPU_R(0.0);
                s.puyOld[particleI] = GPU_R(0.0);
                s.puzOld[particleI] = GPU_R(0.0);
                const GpuReal alphaTheta = clampRange
                (
                    finiteOr(s.thetaDragAlpha[cellI], GPU_R(1.0)),
                    GPU_R(0.0),
                    GPU_R(1.0)
                );
                s.pContactAge[particleI] =
                    clampMin(finiteOr(s.pContactAge[particleI], GPU_R(0.0)), GPU_R(0.0))
                   *alphaTheta;
            }
            particleDiameterM = clampMin
            (
                finiteOr(s.pd[particleI], s.particleDiameterFallback),
                GPU_R(1.0e-12)
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
                GPU_R(1.0e30)
            );
            physicalVolumeM3 =
                (Foam::gpuThermal::finiteContactPi/GPU_R(6.0))
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
                const GpuReal rhoG = clampMin
                (
                    finiteOr(s.couplingRhoOld[cellI], s.rhoMin),
                    s.rhoMin
                );
                const GpuReal ugx = finiteOr(s.couplingUxOld[cellI], GPU_R(0.0));
                const GpuReal ugy = finiteOr(s.couplingUyOld[cellI], GPU_R(0.0));
                const GpuReal ugz = finiteOr(s.couplingUzOld[cellI], GPU_R(0.0));
                const GpuReal re =
                    rhoG*particleDiameterM*sqrt(sqr3(ugx, ugy, ugz))
                   /clampMin(s.gasMu, GPU_R(1.0e-30));
                const GpuReal nusselt =
                    GPU_R(2.0) + GPU_R(0.6)*sqrt(clampMin(re, GPU_R(0.0)))*s.gasPrOneThird;
                const GpuReal particleCp =
                    particleSpecificHeatDevice(particleTemperatureK);
                const GpuReal rate =
                    GPU_R(6.0)*nusselt*molecularGasConductivity(s)
                   /(
                        s.rhoSolid*particleCp
                       *particleDiameterM*particleDiameterM
                      + GPU_TINY(1.0e-300)
                    );
                gasConductanceWK = rate*physicalMassKg*particleCp;
#ifdef GPU_PAIRED_PARTICLE_HEAT
                const double scaledConductance=double(gasConductanceWK)*s.particleHeatCoefficient[cellI];
                gasConductanceWK=static_cast<GpuReal>(scaledConductance);
#endif
            }
            wallEffusivity = s.particleWallEffusivityByFace != nullptr
              ? s.particleWallEffusivityByFace[wallFaceI]
              : sqrt
                (
                    s.particleWallDensityKgM3
                   *s.particleWallSpecificHeatJkgK
                   *s.particleWallConductivityWmK
                );
            maximumAreaM2 = static_cast<GpuReal>
            (
                s.pContactMaximumArea[particleI]
            );
            durationS = static_cast<GpuTime>(s.pContactDuration[particleI]);
            peakTimeFraction = static_cast<GpuReal>
            (
                s.pContactPeakFraction[particleI]
            );
            damageAreaM2 = static_cast<GpuReal>(s.pDepositionArea[particleI]);
            const GpuReal contactAreaScale =
                s.particleWallContactAreaScale[wallFaceI];

            if (finiteContact)
            {
                age0S = clampRange
                (
                    finiteOr(s.pContactAge[particleI], GPU_R(0.0)),
                    GPU_R(0.0),
                    durationS
                );
                thermalDeltaTS = clampRange(dt, GPU_R(0.0), durationS - age0S);
                age1S = age0S + thermalDeltaTS;
                const GpuTime ageMidS = age0S + GPU_R(0.5)*thermalDeltaTS;
                const GpuReal kinematicAreaMidM2 = maximumAreaM2
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
                        static_cast<GpuReal>(s.pCold2DFrozenArea[particleI])
                    ) - damageAreaM2,
                    GPU_R(0.0)
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

        GpuReal meanTemperatureK = particleTemperatureK;
        GpuReal surfaceTemperatureK = particleTemperatureK;
        GpuReal frozenFootprintAreaM2 = static_cast<GpuReal>
        (
            s.pCold2DFrozenArea[particleI]
        );
        GpuReal wallEnergyJ = GPU_R(0.0);
        if (thermalDeltaTS > GPU_R(0.0) && intrinsicAreaM2 > GPU_R(0.0))
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
        GpuReal longDepositAreaM2 = GPU_R(0.0);
        if (radialNode == 0 && finiteContact)
        {
            s.pContactAge[particleI] = age1S;
            const GpuReal theta1 = age1S/durationS;
            const GpuReal kinematicAreaM2 = maximumAreaM2
               *Foam::gpuThermal::normalizedKinematicArea
                (
                    theta1,
                    peakTimeFraction
                );
            const GpuReal effectiveContactAreaM2 =
                fmax(kinematicAreaM2, frozenFootprintAreaM2) - damageAreaM2;
            bool detach = false;
            bool enterLongDeposit = false;
            if
            (
                wallState
             == Foam::gpuThermal::particleWallTransientRebound
            )
            {
                detach = !(effectiveContactAreaM2 > GPU_R(0.0));
                enterLongDeposit =
                    !detach
                 && !(kinematicAreaM2 > GPU_R(0.0))
                 && frozenFootprintAreaM2 > GPU_R(0.0);
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

                const GpuReal targetContactAreaM2 = fmin
                (
                    capillary.equilibriumContactAreaM2,
                    maximumAreaM2
                );
                longDepositAreaM2 =
                    fmax(targetContactAreaM2, frozenFootprintAreaM2)
                   -damageAreaM2;
                detach = !(effectiveContactAreaM2 > GPU_R(0.0));
                enterLongDeposit =
                    !detach
                 && theta1 >= peakTimeFraction
                 &&
                    (
                        kinematicAreaM2 <= targetContactAreaM2
                     || !(kinematicAreaM2 > GPU_R(0.0))
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
                s.pContactAge[particleI] = GPU_R(0.0);
                s.pDepositionArea[particleI] = 0.0;
                s.pContactDuration[particleI] = 0.0;
                s.pContactMaximumArea[particleI] = 0.0;
                s.pContactPeakFraction[particleI] = 0.0;
            }
            else if (transition == 2)
            {
                s.pStuck[particleI] = Foam::gpuThermal::particleWallDeposited;
                s.pContactAge[particleI] = GPU_R(0.0);
                s.pDepositionArea[particleI] =
                    longDepositAreaM2;
                s.puxOld[particleI] = GPU_R(0.0);
                s.puyOld[particleI] = GPU_R(0.0);
                s.puzOld[particleI] = GPU_R(0.0);
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
                s.pCold2DNodeSpecificEnthalpy[nodeBase + axialNode] = 0.0;
            }
            s.pCold2DRingContactAge
            [
                particleI*Foam::gpuThermal::coldWall2DRadialNodeCount
              + radialNode
            ] = 0.0;
            if (radialNode == 0)
            {
                s.pCold2DFrozenArea[particleI] = 0.0;
            }
        }
    }
}

#endif
