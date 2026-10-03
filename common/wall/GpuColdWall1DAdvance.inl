// Shared assembly, nonlinear enthalpy update and publication.
// Precision policies retain the existing cache and stable PCR path.

    const bool inputValid =
        s.coldWallSolidificationEnabled != 0
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
     && gasConductanceWK >= GPU_R(0.0);
    if (!__all_sync(mask, inputValid))
    {
        return false;
    }

    const int nodeBase =
        particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWallRadialRingCount;
    const GpuReal oldEnthalpy = static_cast<GpuReal>
    (
        s.pColdNodeSpecificEnthalpy[nodeBase + lane]
    );
    GpuReal candidateEnthalpy = oldEnthalpy;
    GpuReal guessedTemperature =
        Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (
            oldEnthalpy,
            s.coldWallSolidificationParameters
        );
    GpuReal ringSolidMass = static_cast<GpuReal>
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
    GpuReal frozenArea = __shfl_sync
    (
        mask,
        lane == 0
          ? static_cast<GpuReal>(s.pColdFrozenArea[particleI])
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

    const GpuReal contactArea = Foam::gpuThermal::coldWallClamp
    (
        intrinsicContactAreaM2,
        GPU_REAL_MIN,
        maximumAreaM2
    );
    const GpuReal filmThickness = physicalVolumeM3/contactArea;
    const GpuReal nodeThickness = filmThickness/GPU_R(8.0);
    const GpuReal nodeMass = physicalMassKg/GPU_R(8.0);
    const GpuReal ringArea = maximumAreaM2/GPU_R(8.0);
    const GpuReal ringInnerArea = static_cast<GpuReal>(lane)*ringArea;
    const GpuReal ringWetArea = Foam::gpuThermal::coldWallClamp
    (
        contactArea - ringInnerArea,
        GPU_R(0.0),
        ringArea
    );
    const GpuReal radialFraction = (static_cast<GpuReal>(lane) + GPU_R(0.5))/GPU_R(8.0);
    const GpuReal spreadCoordinate = GPU_R(1.0) - ::sqrt
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
    GpuReal ringCoolingPower = GPU_R(0.0);
    GpuReal acceptedWallPower = GPU_R(0.0);

    for
    (
        int iteration = 0;
        iteration < s.coldWallSolidificationParameters.nonlinearIterations;
        ++iteration
    )
    {

        const GpuReal solidFraction = coldWall1DSolidFractionFromKnownTemperature
        (
            guessedTemperature,
            s.coldWallSolidificationParameters
        );
        const GpuReal conductivity =
            solidFraction*s.coldWallSolidificationParameters.solidThermalConductivityWmK
          + (GPU_R(1.0) - solidFraction)*Foam::gpuThermal::aluminaThermalConductivityWmK;
        const GpuReal bottomConductivity =
            __shfl_sync(mask, conductivity, 0, 8);
        const GpuReal particleResistance =
            GPU_R(0.5)*nodeThickness/bottomConductivity;
#if UGKWP_GPU_REAL_BITS == 32
        const GpuReal ringConductanceIntegral =
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
        const GpuReal ringConductanceIntegral =
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
        const GpuReal ringConductance =
            ringConductanceIntegral/(deltaTSeconds + GPU_REAL_MIN);
        if (!__all_sync(mask, ringConductanceIntegral >= GPU_R(0.0)))
        {
            return false;
        }
        const GpuReal wallConductance =
            coldWall1DGroupSum(ringConductance, mask);
        const GpuReal bottomTemperature =
            __shfl_sync(mask, guessedTemperature, 0, 8);
        ringCoolingPower =
            ringConductance*(bottomTemperature - wallTemperatureK);

        const GpuReal rightConductivity =
            __shfl_down_sync(mask, conductivity, 1, 8);
        const GpuReal faceConductance = lane + 1 < 8
          ? GPU_R(2.0)*conductivity*rightConductivity
           /(conductivity + rightConductivity + GPU_REAL_MIN)
           *contactArea/nodeThickness
          : GPU_R(0.0);
        const GpuReal leftFaceConductance =
            __shfl_up_sync(mask, faceConductance, 1, 8);
        const GpuReal capacity = nodeMass
          *Foam::gpuThermal::coldWallApparentSpecificHeat
            (
                guessedTemperature,
                s.coldWallSolidificationParameters
            )/(deltaTSeconds + GPU_REAL_MIN);
        GpuReal lower = lane > 0 ? -leftFaceConductance : GPU_R(0.0);
        GpuReal diagonal = capacity
          + (lane > 0 ? leftFaceConductance : GPU_R(0.0))
          + (lane + 1 < 8 ? faceConductance : GPU_R(0.0));
        GpuReal upper = lane + 1 < 8 ? -faceConductance : GPU_R(0.0);
        GpuReal rightHandSide = capacity*guessedTemperature;
        if (lane == 0)
        {
            diagonal += wallConductance;
            rightHandSide += wallConductance*wallTemperatureK;
        }
        if (lane == 7)
        {
            diagonal += gasConductanceWK;
            rightHandSide += gasConductanceWK*gasTemperatureK;
        }
#if UGKWP_GPU_REAL_BITS == 32
        const GpuReal solvedTemperature = StabilizePcr
          ? coldWall1DPcrSolveStable
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            capacity + (lane == 0 ? wallConductance : GPU_R(0.0))
              + (lane == 7 ? gasConductanceWK : GPU_R(0.0)),
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
        const GpuReal solvedTemperature = coldWall1DPcrSolve
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            lane,
            mask
        );
#endif
        const GpuReal rightSolvedTemperature =
            __shfl_down_sync(mask, solvedTemperature, 1, 8);
        const GpuReal internalPower = lane + 1 < 8
          ? faceConductance*(solvedTemperature - rightSolvedTemperature)
          : GPU_R(0.0);
        const GpuReal leftInternalPower =
            __shfl_up_sync(mask, internalPower, 1, 8);
        const GpuReal wallPower = wallConductance
          *(__shfl_sync(mask, solvedTemperature, 0, 8) - wallTemperatureK);
        const GpuReal gasPower = gasConductanceWK
          *(gasTemperatureK - __shfl_sync(mask, solvedTemperature, 7, 8));
        GpuReal power =
            (lane > 0 ? leftInternalPower : GPU_R(0.0))
          - (lane + 1 < 8 ? internalPower : GPU_R(0.0));
        if (lane == 0)
        {
            power -= wallPower;
        }
        if (lane == 7)
        {
            power += gasPower;
        }
        candidateEnthalpy = oldEnthalpy
          + deltaTSeconds*power/nodeMass;
        guessedTemperature =
            Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
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
