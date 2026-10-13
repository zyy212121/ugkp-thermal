#pragma once
#include <type_traits>

// No contact geometry is constructed here. Gas exchange spans the physical
// step even when finite contact ends or the midpoint footprint vanishes.
__device__ inline bool advanceColdWall1DGasOnlyGroup
(
    DeviceState& s,
    const int particleI,
    const int lane,
    const unsigned int mask,
    const GpuReal physicalMassKg,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK,
    const GpuTime deltaTSeconds,
    const GpuTime activeContactDeltaTSeconds,
    GpuReal& meanTemperatureK
)
{
    using namespace Foam::gpuThermal;
    const bool inputValid = true
#ifdef GPU_PAIRED_PARTICLE_HEAT
     && s.pThermalEnthalpyResidual != nullptr
#endif
     && s.pColdNodeSpecificEnthalpy != nullptr
     && s.pColdRingSolidMass != nullptr
     && s.pColdFrozenArea != nullptr
     && s.pColdContactAge != nullptr;
    if (!__all_sync(mask, inputValid))
    {
        return false;
    }
    const int nodeBase = particleI*coldWallAxialNodeCount;
    const int ringBase = particleI*coldWallRadialRingCount;
#ifdef GPU_PAIRED_PARTICLE_HEAT
    // Gas heat has already been paired and uniformly published. This route
    // only refreshes derived phase/history; zero heat never rewrites nodes.
    const double carry = s.pThermalEnthalpyResidual[particleI];
    const double effectiveH = double(s.pColdNodeSpecificEnthalpy[nodeBase+lane])+carry;
    const double temperature = coldWallCarryTemperature(effectiveH,s.coldWallSolidificationParameters);
    const double meanH = coldWall1DGroupSum(double(s.pColdNodeSpecificEnthalpy[nodeBase+lane])/8.0,mask)+carry;
    const double fraction = coldWall1DGroupMinPrefix(coldWallCarryFraction(temperature,s.coldWallSolidificationParameters),lane,mask);
    const double connectedMass = coldWall1DGroupSum(double(physicalMassKg)*fraction/8.0,mask);
    const double assignedMass = coldWall1DGroupSum(double(s.pColdRingSolidMass[ringBase+lane]),mask);
    const GpuTime oldAge = s.pColdContactAge[particleI];
    const GpuTime nextAge = oldAge+activeContactDeltaTSeconds;
    // Every lane finishes its shared age load before lane zero publishes it.
    __syncwarp(mask);
    if(connectedMass<assignedMass && assignedMass>0.0)
      s.pColdRingSolidMass[ringBase+lane]=double(s.pColdRingSolidMass[ringBase+lane])*connectedMass/assignedMass;
    if(lane==0)s.pColdContactAge[particleI]=nextAge;
    meanTemperatureK=static_cast<GpuReal>(coldWallCarryTemperature(meanH,s.coldWallSolidificationParameters));
    (void)gasTemperatureK;(void)gasConductanceWK;
    return true;
#else
    const GpuReal oldEnthalpy = static_cast<GpuReal>
    (
        s.pColdNodeSpecificEnthalpy[nodeBase + lane]
    );
    GpuReal ringSolidMass = static_cast<GpuReal>(s.pColdRingSolidMass[ringBase + lane]);
    const GpuTime oldAge = __shfl_sync
    (
        mask, lane == 0 ? static_cast<GpuTime>(s.pColdContactAge[particleI]) : 0.0, 0, 8
    );
    const GpuTime newAge = oldAge + activeContactDeltaTSeconds;
    const GpuReal meanOldEnthalpy = coldWall1DGroupSum(oldEnthalpy/GPU_R(8.0), mask);
    GpuReal gasIncrement = GPU_R(0.0);
    if (lane == 0)
    {
        coldWallGasSpecificEnthalpyIncrement
        (
            meanOldEnthalpy, physicalMassKg, gasTemperatureK, gasConductanceWK,
            deltaTSeconds, s.coldWallSolidificationParameters, gasIncrement
        );
    }

    gasIncrement = __shfl_sync(mask, gasIncrement, 0, 8);
    const GpuReal candidateEnthalpy = oldEnthalpy + gasIncrement;
    const auto storedEnthalpy = static_cast<typename std::remove_reference<decltype(s.pColdNodeSpecificEnthalpy[0])>::type>(candidateEnthalpy);
    // A completed but pinned contact can persist as a deposit. Retain its
    // footprint history, but apply the existing melt-down scaling to ring
    // masses; no new solid mass or pinning is assigned without a wet geometry.
    if (gasIncrement > GPU_R(0.0))
    {
        GpuReal connectedFraction = coldWallSolidFraction
        (
            static_cast<GpuReal>(storedEnthalpy), s.coldWallSolidificationParameters
        );
        connectedFraction = coldWall1DGroupMinPrefix(connectedFraction, lane, mask);
        const GpuReal connectedMass = coldWall1DGroupSum
        (
            physicalMassKg/GPU_R(8.0)*connectedFraction, mask
        );
        const GpuReal assignedMass = coldWall1DGroupSum(ringSolidMass, mask);

        if (connectedMass < assignedMass && assignedMass > GPU_R(0.0))
        {
            ringSolidMass *= connectedMass/assignedMass;
        }
    }

    const GpuReal meanEnthalpy = coldWall1DGroupSum
    (
        static_cast<GpuReal>(storedEnthalpy)/GPU_R(8.0), mask
    );
    GpuReal meanTemperature = GPU_R(0.0);
    if (lane == 0)
    {
        meanTemperature = coldWallTemperatureFromSpecificEnthalpyK
        (
            meanEnthalpy, s.coldWallSolidificationParameters
        );
    }
    meanTemperature = __shfl_sync(mask, meanTemperature, 0, 8);

    // Publish the complete result after required arithmetic has finished.
    s.pColdNodeSpecificEnthalpy[nodeBase + lane] = storedEnthalpy;
    s.pColdRingSolidMass[ringBase + lane] = ringSolidMass;
    if (lane == 0)
    {
        s.pColdContactAge[particleI] = newAge;
    }
    meanTemperatureK = meanTemperature;
    return true;
#endif
}
