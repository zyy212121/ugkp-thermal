    GpuReal connectedFraction = coldWall1DSolidFractionFromKnownTemperature
    (
        guessedTemperature,
        s.coldWallSolidificationParameters
    );
    connectedFraction = coldWall1DGroupMinPrefix
    (
        connectedFraction,
        lane,
        mask
    );
    const GpuReal connectedMass = coldWall1DGroupSum
    (
        nodeMass*connectedFraction,
        mask
    );
    GpuReal assignedMass = coldWall1DGroupSum(ringSolidMass, mask);
    if (connectedMass < assignedMass && assignedMass > GPU_R(0.0))
    {
        ringSolidMass *= connectedMass/assignedMass;
        assignedMass = connectedMass;
    }
    GpuReal remainingMass = connectedMass - assignedMass;
    const GpuReal ringCapacity =
        s.coldWallSolidificationParameters.solidDensityKgM3
       *ringWetArea*filmThickness;
    for
    (
        int pass = 0;
        pass < 8 && remainingMass > GPU_R(1.0e-18)*physicalMassKg;
        ++pass
    )
    {
        const GpuReal available = ringCapacity - ringSolidMass;
        const GpuReal cooling = ringCoolingPower > GPU_R(0.0)
          ? ringCoolingPower
          : GPU_R(0.0);
        const GpuReal weight = available > GPU_R(0.0)
          ? (cooling > GPU_R(0.0) ? cooling : ringWetArea)
          : GPU_R(0.0);
        const GpuReal weightSum = coldWall1DGroupSum(weight, mask);
        if (!(weightSum > GPU_R(0.0)))
        {
            break;
        }
        const GpuReal requested = remainingMass*weight/weightSum;
        const GpuReal addition = available > GPU_R(0.0)
          ? (requested < available ? requested : available)
          : GPU_R(0.0);
        ringSolidMass += addition;
        const GpuReal allocated = coldWall1DGroupSum(addition, mask);
        remainingMass -= allocated;
        if (!(allocated > GPU_R(0.0)))
        {
            break;
        }
    }

    const GpuReal localSolidFraction = ringWetArea > GPU_R(0.0)
      ? ringSolidMass
       /(s.coldWallSolidificationParameters.solidDensityKgM3
        *ringWetArea*filmThickness)
      : GPU_R(0.0);
    const unsigned int eligibleMask = __ballot_sync
    (
        mask,
        ringWetArea > GPU_R(0.0)
     && localSolidFraction
        >= s.coldWallSolidificationParameters.pinningThicknessFraction
    ) >> (__ffs(mask) - 1);
    const unsigned int lowerMask = (1u << (lane + 1)) - 1u;
    const bool connectedRing =
        (eligibleMask & lowerMask) == lowerMask;
    const GpuReal candidateFrozenArea = coldWall1DGroupSum
    (
        connectedRing ? ringWetArea : GPU_R(0.0),
        mask
    );
    if (candidateFrozenArea > frozenArea)
    {
        frozenArea = candidateFrozenArea;
    }
    profileContactAgeS += deltaTSeconds;

    s.pColdNodeSpecificEnthalpy[nodeBase + lane] =
        static_cast<float>(candidateEnthalpy);
    s.pColdRingSolidMass[ringBase + lane] =
        static_cast<float>(ringSolidMass);
    if (lane == 0)
    {
        s.pColdContactAge[particleI] =
            static_cast<GpuTime>(profileContactAgeS);
        s.pColdFrozenArea[particleI] = static_cast<float>(frozenArea);
    }
    const GpuReal meanEnthalpy = coldWall1DGroupSum(candidateEnthalpy, mask)/GPU_R(8.0);
    GpuReal groupMeanTemperature = GPU_R(0.0);
    if (lane == 0)
    {
        groupMeanTemperature =
            Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
            (
                meanEnthalpy,
                s.coldWallSolidificationParameters
            );
    }
    meanTemperatureK = __shfl_sync(mask, groupMeanTemperature, 0, 8);
    frozenFootprintAreaM2 = frozenArea;
    wallEnergyJ = acceptedWallPower*deltaTSeconds;
    stateValid =
        Foam::gpuThermal::finiteColdWallValue(meanTemperatureK)
     && meanTemperatureK > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(frozenFootprintAreaM2)
     && frozenFootprintAreaM2 >= GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(wallEnergyJ);
    return __all_sync(mask, stateValid);
