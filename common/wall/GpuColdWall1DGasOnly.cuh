#pragma once

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
    const bool inputValid = s.coldWallSolidificationEnabled != 0
     && s.pColdNodeSpecificEnthalpy != nullptr
     && s.pColdRingSolidMass != nullptr
     && s.pColdFrozenArea != nullptr
     && s.pColdContactAge != nullptr
     && validColdWallSolidificationParameters(s.coldWallSolidificationParameters)
     && finiteColdWallValue(physicalMassKg) && physicalMassKg > GPU_R(0.0)
     && deltaTSeconds >= 0.0 && deltaTSeconds <= DBL_MAX
     && activeContactDeltaTSeconds >= 0.0
     && activeContactDeltaTSeconds <= deltaTSeconds;
    if (!__all_sync(mask, inputValid))
    {
        return false;
    }
    const int nodeBase = particleI*coldWallAxialNodeCount;
    const int ringBase = particleI*coldWallRadialRingCount;
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
    const GpuReal frozenArea = static_cast<GpuReal>(s.pColdFrozenArea[particleI]);
    const bool oldStateValid = finiteColdWallValue(oldEnthalpy) && oldEnthalpy > GPU_R(0.0)
     && finiteColdWallValue(ringSolidMass) && ringSolidMass >= GPU_R(0.0)
     && finiteColdWallValue(frozenArea) && frozenArea >= GPU_R(0.0)
     && oldAge >= 0.0 && oldAge <= DBL_MAX
     && newAge >= oldAge && newAge <= DBL_MAX;
    if (!__all_sync(mask, oldStateValid))
    {
        return false;
    }

    const GpuReal meanOldEnthalpy = coldWall1DGroupSum(oldEnthalpy/GPU_R(8.0), mask);
    GpuReal gasIncrement = GPU_R(0.0);
    bool sourceValid = true;
    if (lane == 0)
    {
        sourceValid = coldWallGasSpecificEnthalpyIncrement
        (
            meanOldEnthalpy, physicalMassKg, gasTemperatureK, gasConductanceWK,
            deltaTSeconds, s.coldWallSolidificationParameters, gasIncrement
        );
    }
    if (!__all_sync(mask, sourceValid))
    {
        return false;
    }
    gasIncrement = __shfl_sync(mask, gasIncrement, 0, 8);
    const GpuReal candidateEnthalpy = oldEnthalpy + gasIncrement;
    const float storedEnthalpy = static_cast<float>(candidateEnthalpy);
    const bool candidateValid = finiteColdWallValue(candidateEnthalpy)
     && candidateEnthalpy > GPU_R(0.0)
     && finiteColdWallValue(static_cast<GpuReal>(storedEnthalpy))
     && storedEnthalpy > 0.0f;
    if (!__all_sync(mask, candidateValid))
    {
        return false;
    }

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
        if (!__all_sync(mask, finiteColdWallValue(assignedMass)))
        {
            return false;
        }
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
    if (!__all_sync(mask, finiteColdWallValue(meanTemperature)
        && meanTemperature > GPU_R(0.0) && finiteColdWallValue(ringSolidMass)
        && ringSolidMass >= GPU_R(0.0)))
    {
        return false;
    }

    // Publish only after every lane and the aggregate output are admissible.
    s.pColdNodeSpecificEnthalpy[nodeBase + lane] = storedEnthalpy;
    s.pColdRingSolidMass[ringBase + lane] = static_cast<float>(ringSolidMass);
    if (lane == 0)
    {
        s.pColdContactAge[particleI] = newAge;
    }
    meanTemperatureK = meanTemperature;
    return true;
}
