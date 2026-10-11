// Shared assembly, nonlinear enthalpy update and publication.
// Precision policies retain the existing cache and stable PCR path.

    const bool inputValid =
        s.coldWallSolidificationEnabled != 0
#ifdef GPU_PAIRED_PARTICLE_HEAT
     && s.pThermalEnthalpyResidual != nullptr
     && s.pm[particleI] > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(s.pm[particleI])
#endif
     && s.pColdNodeSpecificEnthalpy != nullptr
     && s.pColdRingSolidMass != nullptr
     && s.pColdFrozenArea != nullptr
     && s.pColdContactAge != nullptr
     && Foam::gpuThermal::validColdWallSolidificationParameters
        (
            s.coldWallSolidificationParameters
        )
     && Foam::gpuThermal::finiteColdWallValue(physicalVolumeM3)
     && physicalVolumeM3 > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(physicalMassKg)
     && physicalMassKg > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(maximumAreaM2)
     && maximumAreaM2 > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(intrinsicContactAreaM2)
     && intrinsicContactAreaM2 > GPU_R(0.0)
     && intrinsicContactAreaM2 <= maximumAreaM2
     && Foam::gpuThermal::finiteColdWallValue(contactDurationS)
     && contactDurationS > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(peakTimeFraction)
     && peakTimeFraction > GPU_R(0.0)
     && peakTimeFraction < GPU_R(1.0)
     && Foam::gpuThermal::finiteColdWallValue(deltaTSeconds)
     && deltaTSeconds >= GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(wallTemperatureK)
     && wallTemperatureK > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(wallEffusivity)
     && wallEffusivity > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(thermalAreaFactor)
     && thermalAreaFactor > GPU_R(0.0)
     && thermalAreaFactor <= GPU_R(1.0)
     && Foam::gpuThermal::finiteColdWallValue(gasTemperatureK)
     && gasTemperatureK > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(gasConductanceWK)
     && gasConductanceWK >= GPU_R(0.0)
     && (gasExchangeDurationS == -1.0
       || (gasExchangeDurationS >= 0.0 && gasExchangeDurationS <= DBL_MAX));
    if (!__all_sync(mask, inputValid))
    {
        return false;
    }

    const int nodeBase =
        particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWallRadialRingCount;
    const ColdWallThermalReal storedOldEnthalpy = static_cast<ColdWallThermalReal>
    (
        s.pColdNodeSpecificEnthalpy[nodeBase + lane]
    );
#ifdef GPU_PAIRED_PARTICLE_HEAT
    const double oldCarry = s.pThermalEnthalpyResidual[particleI];
    const double oldEnthalpy = storedOldEnthalpy + oldCarry;
#else
    const GpuReal oldEnthalpy = storedOldEnthalpy;
#endif
    ColdWallThermalReal candidateEnthalpy = oldEnthalpy;
    ColdWallThermalReal guessedTemperature =
        coldWallCarryTemperature
        (
            oldEnthalpy,
            s.coldWallSolidificationParameters
        );
    ColdWallThermalReal ringSolidMass = static_cast<ColdWallThermalReal>
    (
        s.pColdRingSolidMass[ringBase + lane]
    );
    GpuTime profileContactAgeS = __shfl_sync
    (
        mask,
        lane == 0
          ? static_cast<GpuTime>(s.pColdContactAge[particleI])
          : GPU_R(0.0),
        0,
        8
    );
    ColdWallThermalReal frozenArea = __shfl_sync
    (
        mask,
        lane == 0
          ? static_cast<ColdWallThermalReal>(s.pColdFrozenArea[particleI])
          : GPU_R(0.0),
        0,
        8
    );
    int stateValid =
        Foam::gpuThermal::finiteColdWallValue(oldEnthalpy)
     && oldEnthalpy >= GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(guessedTemperature)
     && guessedTemperature > GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(ringSolidMass)
     && ringSolidMass >= GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(profileContactAgeS)
     && profileContactAgeS >= GPU_R(0.0)
     && Foam::gpuThermal::finiteColdWallValue(frozenArea)
     && frozenArea >= GPU_R(0.0);
    if (!__all_sync(mask, stateValid))
    {
        return false;
    }

    [[maybe_unused]] const ColdWallThermalReal meanOldEnthalpy = coldWall1DGroupSum(oldEnthalpy/GPU_R(8.0), mask);
    ColdWallThermalReal gasSpecificEnthalpyIncrement = GPU_R(0.0);
    [[maybe_unused]] const GpuTime gasDeltaTSeconds = gasExchangeDurationS == -1.0
      ? deltaTSeconds : gasExchangeDurationS;
    bool gasSourceValid = true;
#ifndef GPU_PAIRED_PARTICLE_HEAT
    if (lane == 0)
    {
        gasSourceValid = Foam::gpuThermal::coldWallGasSpecificEnthalpyIncrement
        (
            meanOldEnthalpy, physicalMassKg, gasTemperatureK, gasConductanceWK,
            gasDeltaTSeconds, s.coldWallSolidificationParameters,
            gasSpecificEnthalpyIncrement
        );
    }
#endif
    if (!__all_sync(mask, gasSourceValid))
    {
        return false;
    }
    gasSpecificEnthalpyIncrement = __shfl_sync(mask, gasSpecificEnthalpyIncrement, 0, 8);

#ifdef GPU_PAIRED_PARTICLE_HEAT
    if (deltaTSeconds == 0.0) {
        meanTemperatureK = static_cast<GpuReal>(coldWallCarryTemperature(meanOldEnthalpy,s.coldWallSolidificationParameters));
        frozenFootprintAreaM2 = frozenArea; wallEnergyJ = GPU_R(0.0);
        return true;
    }
#endif
    const ColdWallThermalReal contactArea = Foam::gpuThermal::coldWallClamp
    (
        intrinsicContactAreaM2,
        GPU_REAL_MIN,
        maximumAreaM2
    );
    const ColdWallThermalReal filmThickness = physicalVolumeM3/contactArea;
    const ColdWallThermalReal nodeThickness = filmThickness/GPU_R(8.0);
    const ColdWallThermalReal nodeMass = physicalMassKg/GPU_R(8.0);
    const ColdWallThermalReal nodeGasPower = nodeMass*gasSpecificEnthalpyIncrement
      /(deltaTSeconds + GPU_REAL_MIN);
    const ColdWallThermalReal ringArea = maximumAreaM2/GPU_R(8.0);
    const ColdWallThermalReal ringInnerArea = static_cast<ColdWallThermalReal>(lane)*ringArea;
    const ColdWallThermalReal ringWetArea = Foam::gpuThermal::coldWallClamp
    (
        contactArea - ringInnerArea,
        GPU_R(0.0),
        ringArea
    );
    const ColdWallThermalReal radialFraction = (static_cast<ColdWallThermalReal>(lane) + GPU_R(0.5))/GPU_R(8.0);
    const ColdWallThermalReal spreadCoordinate = GPU_R(1.0) - ::sqrt
    (
        Foam::gpuThermal::coldWallClamp(GPU_R(1.0) - radialFraction, GPU_R(0.0), GPU_R(1.0))
    );
    const GpuTime firstWetAge =
        contactDurationS*peakTimeFraction*spreadCoordinate;
    const GpuTime profileContactAge1S = profileContactAgeS + deltaTSeconds;
    const GpuTime localAge0 = profileContactAgeS > firstWetAge
      ? profileContactAgeS - firstWetAge
      : GPU_R(0.0);
    const GpuTime localAge1 = profileContactAge1S > firstWetAge
      ? profileContactAge1S - firstWetAge
      : GPU_R(0.0);
#if UGKWP_GPU_REAL_BITS == 32
    const ColdWallIntegralCache integralCache = coldWallPrepareIntegralCache
    (
        localAge0, localAge1, wallEffusivity,
        s.coldWallSolidificationParameters.wallTransientResistance != 0
    );
#endif
    ColdWallThermalReal ringCoolingPower = GPU_R(0.0);
    ColdWallThermalReal acceptedWallPower = GPU_R(0.0);

    for
    (
        int iteration = 0;
        iteration < s.coldWallSolidificationParameters.nonlinearIterations;
        ++iteration
    )
    {

        const ColdWallThermalReal solidFraction = coldWall1DSolidFractionFromKnownTemperature
        (
            guessedTemperature,
            s.coldWallSolidificationParameters
        );
        const ColdWallThermalReal conductivity =
            solidFraction*s.coldWallSolidificationParameters.solidThermalConductivityWmK
          + (GPU_R(1.0) - solidFraction)*Foam::gpuThermal::aluminaThermalConductivityWmK;
        const ColdWallThermalReal bottomConductivity =
            __shfl_sync(mask, conductivity, 0, 8);
        const ColdWallThermalReal particleResistance =
            GPU_R(0.5)*nodeThickness/bottomConductivity;
#if UGKWP_GPU_REAL_BITS == 32
        const ColdWallThermalReal ringConductanceIntegral =
            coldWallCachedConductanceTimeIntegral
            (
                integralCache,
                ringWetArea,
                thermalAreaFactor,
                localAge0,
                localAge1,
                s.coldWallSolidificationParameters.interfaceResistanceM2KW,
                particleResistance,
                wallEffusivity,
                s.coldWallSolidificationParameters.wallTransientResistance != 0
            );
#else
        const ColdWallThermalReal ringConductanceIntegral =
            Foam::gpuThermal::wallInterfaceConductanceTimeIntegral
            (
                ringWetArea,
                thermalAreaFactor,
                localAge0,
                localAge1,
                s.coldWallSolidificationParameters.interfaceResistanceM2KW,
                particleResistance,
                wallEffusivity,
                s.coldWallSolidificationParameters.wallTransientResistance != 0
            );
#endif
        const ColdWallThermalReal ringConductance =
            ringConductanceIntegral/(deltaTSeconds + GPU_REAL_MIN);
        if (!__all_sync(mask, ringConductanceIntegral >= GPU_R(0.0)))
        {
            return false;
        }
        const ColdWallThermalReal wallConductance =
            coldWall1DGroupSum(ringConductance, mask);
        const ColdWallThermalReal bottomTemperature =
            __shfl_sync(mask, guessedTemperature, 0, 8);
        ringCoolingPower =
            ringConductance*(bottomTemperature - wallTemperatureK);

        const ColdWallThermalReal rightConductivity =
            __shfl_down_sync(mask, conductivity, 1, 8);
        const ColdWallThermalReal faceConductance = lane + 1 < 8
          ? GPU_R(2.0)*conductivity*rightConductivity
           /(conductivity + rightConductivity + GPU_REAL_MIN)
           *contactArea/nodeThickness
          : GPU_R(0.0);
        const ColdWallThermalReal leftFaceConductance =
            __shfl_up_sync(mask, faceConductance, 1, 8);
        const ColdWallThermalReal capacity = nodeMass
          *coldWallCarrySpecificHeat
            (
                guessedTemperature,
                s.coldWallSolidificationParameters
            )/(deltaTSeconds + GPU_REAL_MIN);
        ColdWallThermalReal lower = lane > 0 ? -leftFaceConductance : GPU_R(0.0);
        ColdWallThermalReal diagonal = capacity
          + (lane > 0 ? leftFaceConductance : GPU_R(0.0))
          + (lane + 1 < 8 ? faceConductance : GPU_R(0.0));
        ColdWallThermalReal upper = lane + 1 < 8 ? -faceConductance : GPU_R(0.0);
        // candidateEnthalpy is H_guess, not another physical time step.
        ColdWallThermalReal rightHandSide = capacity*guessedTemperature
          + nodeMass*(oldEnthalpy - candidateEnthalpy)
           /(deltaTSeconds + GPU_REAL_MIN)
          + nodeGasPower;
        if (lane == 0)
        {
            diagonal += wallConductance;
            rightHandSide += wallConductance*wallTemperatureK;
        }
#if UGKWP_GPU_REAL_BITS == 32
        const ColdWallThermalReal solvedTemperature = StabilizePcr
          ? coldWall1DPcrSolveStable
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            capacity + (lane == 0 ? wallConductance : GPU_R(0.0)),
            lane,
            mask
        )
          : coldWall1DPcrSolve
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            lane,
            mask
        );

#else
        const ColdWallThermalReal solvedTemperature = coldWall1DPcrSolve
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            lane,
            mask
        );
#endif
        const ColdWallThermalReal rightSolvedTemperature =
            __shfl_down_sync(mask, solvedTemperature, 1, 8);
        const ColdWallThermalReal internalPower = lane + 1 < 8
          ? faceConductance*(solvedTemperature - rightSolvedTemperature)
          : GPU_R(0.0);
        const ColdWallThermalReal leftInternalPower =
            __shfl_up_sync(mask, internalPower, 1, 8);
        const ColdWallThermalReal wallPower = wallConductance
          *(__shfl_sync(mask, solvedTemperature, 0, 8) - wallTemperatureK);
        ColdWallThermalReal power =
            (lane > 0 ? leftInternalPower : GPU_R(0.0))
          - (lane + 1 < 8 ? internalPower : GPU_R(0.0));
        if (lane == 0)
        {
            power -= wallPower;
        }
        candidateEnthalpy = oldEnthalpy
          + gasSpecificEnthalpyIncrement
          + deltaTSeconds*power/nodeMass;
        guessedTemperature =
            coldWallCarryTemperature
            (
                candidateEnthalpy,
                s.coldWallSolidificationParameters
            );
        acceptedWallPower = wallPower;
        stateValid =
            Foam::gpuThermal::finiteColdWallValue(candidateEnthalpy)
         && candidateEnthalpy >= GPU_R(0.0)
         && Foam::gpuThermal::finiteColdWallValue(guessedTemperature)
         && guessedTemperature > GPU_R(0.0);
        if (!__all_sync(mask, stateValid))
        {
#if UGKWP_GPU_REAL_BITS == 32
            if (linearSolveFailed != nullptr) *linearSolveFailed = true;
#endif
            return false;
        }
    }

#include "GpuColdWall1DPublish.inl"
