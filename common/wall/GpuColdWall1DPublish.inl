#ifdef GPU_PAIRED_PARTICLE_HEAT
    // Close against the exact value the existing wall-ledger call publishes.
    wallEnergyJ = static_cast<GpuReal>(acceptedWallPower*deltaTSeconds);
    const GpuReal multiplicity = s.pm[particleI]/physicalMassKg;
    const GpuReal representedWallEnergy = multiplicity*wallEnergyJ;
    const double meanIncrement = gasSpecificEnthalpyIncrement
      - double(representedWallEnergy)/double(s.pm[particleI]);
    const double candidateMean = coldWall1DGroupSum(candidateEnthalpy/8.0,mask);
    const double meanCorrection = (meanOldEnthalpy-candidateMean)+meanIncrement;
    candidateEnthalpy += meanCorrection;
    const float publishedEnthalpy = ugkwp::coldWallStoredEnthalpyFloor(candidateEnthalpy);
    const double storageChange = coldWall1DGroupSum
      ((double(publishedEnthalpy)-storedOldEnthalpy)/8.0,mask);
    const double nextCarry = (oldCarry-storageChange)+meanIncrement;
    guessedTemperature = coldWallCarryTemperature(double(publishedEnthalpy)+nextCarry,
      s.coldWallSolidificationParameters);
#endif
    ColdWallThermalReal connectedFraction = coldWall1DSolidFractionFromKnownTemperature
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
    const ColdWallThermalReal connectedMass = coldWall1DGroupSum
    (
        nodeMass*connectedFraction,
        mask
    );
    ColdWallThermalReal assignedMass = coldWall1DGroupSum(ringSolidMass, mask);
    if (connectedMass < assignedMass && assignedMass > GPU_R(0.0))
    {
        ringSolidMass *= connectedMass/assignedMass;
        assignedMass = connectedMass;
    }
    ColdWallThermalReal remainingMass = connectedMass - assignedMass;
    const ColdWallThermalReal ringCapacity =
        s.coldWallSolidificationParameters.solidDensityKgM3
       *ringWetArea*filmThickness;
    for
    (
        int pass = 0;
        pass < 8 && remainingMass > GPU_R(1.0e-18)*physicalMassKg;
        ++pass
    )
    {
        const ColdWallThermalReal available = ringCapacity - ringSolidMass;
        const ColdWallThermalReal cooling = ringCoolingPower > GPU_R(0.0)
          ? ringCoolingPower
          : GPU_R(0.0);
        const ColdWallThermalReal weight = available > GPU_R(0.0)
          ? (cooling > GPU_R(0.0) ? cooling : ringWetArea)
          : GPU_R(0.0);
        const ColdWallThermalReal weightSum = coldWall1DGroupSum(weight, mask);
        if (!(weightSum > GPU_R(0.0)))
        {
            break;
        }
        const ColdWallThermalReal requested = remainingMass*weight/weightSum;
        const ColdWallThermalReal addition = available > GPU_R(0.0)
          ? (requested < available ? requested : available)
          : GPU_R(0.0);
        ringSolidMass += addition;
        const ColdWallThermalReal allocated = coldWall1DGroupSum(addition, mask);
        remainingMass -= allocated;
        if (!(allocated > GPU_R(0.0)))
        {
            break;
        }
    }

    const ColdWallThermalReal localSolidFraction = ringWetArea > GPU_R(0.0)
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
    const ColdWallThermalReal candidateFrozenArea = coldWall1DGroupSum
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
#ifdef GPU_PAIRED_PARTICLE_HEAT
        publishedEnthalpy;
#else
        candidateEnthalpy;
#endif
    s.pColdRingSolidMass[ringBase + lane] =
        ringSolidMass;
    if (lane == 0)
    {
        s.pColdContactAge[particleI] =
            static_cast<GpuTime>(profileContactAgeS);
        s.pColdFrozenArea[particleI] = frozenArea;
    }
#ifdef GPU_PAIRED_PARTICLE_HEAT
    if (lane == 0) s.pThermalEnthalpyResidual[particleI] = nextCarry;
    const double meanEnthalpy = coldWall1DGroupSum(double(publishedEnthalpy)/8.0,mask)+nextCarry;
#else
    const GpuReal meanEnthalpy = coldWall1DGroupSum(candidateEnthalpy, mask)/GPU_R(8.0);
#endif
    ColdWallThermalReal groupMeanTemperature = GPU_R(0.0);
    if (lane == 0)
    {
        groupMeanTemperature =
            coldWallCarryTemperature
            (
                meanEnthalpy,
                s.coldWallSolidificationParameters
            );
    }
    meanTemperatureK = __shfl_sync(mask, groupMeanTemperature, 0, 8);
    frozenFootprintAreaM2 = frozenArea;
    wallEnergyJ = acceptedWallPower*deltaTSeconds;
    return true;
