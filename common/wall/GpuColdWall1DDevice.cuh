#ifndef GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH

__device__ inline double coldWall1DGroupSum
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

__device__ inline double coldWall1DGroupMinPrefix
(
    double value,
    const int lane,
    const unsigned int mask
)
{
    for (int offset = 1; offset < 8; offset *= 2)
    {
        const double lower = __shfl_up_sync(mask, value, offset, 8);
        if (lane >= offset && lower < value)
        {
            value = lower;
        }
    }
    return value;
}

__device__ inline double coldWall1DPcrSolve
(
    double lower,
    double diagonal,
    double upper,
    double rightHandSide,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const double lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const double lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const double lowerUpper =
            __shfl_up_sync(mask, upper, stride, 8);
        const double lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const double upperLower =
            __shfl_down_sync(mask, lower, stride, 8);
        const double upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const double upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const double upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const double alpha = lane >= stride
          ? -lower/(lowerDiagonal + DBL_MIN)
          : 0.0;
        const double beta = lane + stride < 8
          ? -upper/(upperDiagonal + DBL_MIN)
          : 0.0;
        const double nextLower = lane >= stride
          ? alpha*lowerLower
          : 0.0;
        const double nextUpper = lane + stride < 8
          ? beta*upperUpper
          : 0.0;
        const double nextDiagonal = diagonal
          + alpha*lowerUpper + beta*upperLower;
        const double nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + DBL_MIN);
}

__device__ bool advanceColdWall1DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int lane,
    const unsigned int mask,
    const double physicalVolumeM3,
    const double physicalMassKg,
    const double maximumAreaM2,
    const double intrinsicContactAreaM2,
    const double contactDurationS,
    const double peakTimeFraction,
    const double deltaTSeconds,
    const double wallTemperatureK,
    const double wallEffusivity,
    const double thermalAreaFactor,
    const double gasTemperatureK,
    const double gasConductanceWK,
    double& meanTemperatureK,
    double& frozenFootprintAreaM2,
    double& wallEnergyJ
)
{
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
     && physicalVolumeM3 > 0.0
     && Foam::gpuThermal::finiteColdWallValue(physicalMassKg)
     && physicalMassKg > 0.0
     && Foam::gpuThermal::finiteColdWallValue(maximumAreaM2)
     && maximumAreaM2 > 0.0
     && Foam::gpuThermal::finiteColdWallValue(intrinsicContactAreaM2)
     && intrinsicContactAreaM2 > 0.0
     && intrinsicContactAreaM2 <= maximumAreaM2
     && Foam::gpuThermal::finiteColdWallValue(contactDurationS)
     && contactDurationS > 0.0
     && Foam::gpuThermal::finiteColdWallValue(peakTimeFraction)
     && peakTimeFraction > 0.0
     && peakTimeFraction < 1.0
     && Foam::gpuThermal::finiteColdWallValue(deltaTSeconds)
     && deltaTSeconds >= 0.0
     && Foam::gpuThermal::finiteColdWallValue(wallTemperatureK)
     && wallTemperatureK > 0.0
     && Foam::gpuThermal::finiteColdWallValue(wallEffusivity)
     && wallEffusivity > 0.0
     && Foam::gpuThermal::finiteColdWallValue(thermalAreaFactor)
     && thermalAreaFactor > 0.0
     && thermalAreaFactor <= 1.0
     && Foam::gpuThermal::finiteColdWallValue(gasTemperatureK)
     && gasTemperatureK > 0.0
     && Foam::gpuThermal::finiteColdWallValue(gasConductanceWK)
     && gasConductanceWK >= 0.0;
    if (!__all_sync(mask, inputValid))
    {
        return false;
    }

    const int nodeBase =
        particleI*Foam::gpuThermal::coldWallAxialNodeCount;
    const int ringBase =
        particleI*Foam::gpuThermal::coldWallRadialRingCount;
    const double oldEnthalpy = static_cast<double>
    (
        s.pColdNodeSpecificEnthalpy[nodeBase + lane]
    );
    double candidateEnthalpy = oldEnthalpy;
    double guessedTemperature =
        Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (
            oldEnthalpy,
            s.coldWallSolidificationParameters
        );
    double ringSolidMass = static_cast<double>
    (
        s.pColdRingSolidMass[ringBase + lane]
    );
    double profileContactAgeS = __shfl_sync
    (
        mask,
        lane == 0
          ? static_cast<double>(s.pColdContactAge[particleI])
          : 0.0,
        0,
        8
    );
    double frozenArea = __shfl_sync
    (
        mask,
        lane == 0
          ? static_cast<double>(s.pColdFrozenArea[particleI])
          : 0.0,
        0,
        8
    );
    int stateValid =
        Foam::gpuThermal::finiteColdWallValue(oldEnthalpy)
     && oldEnthalpy >= 0.0
     && Foam::gpuThermal::finiteColdWallValue(guessedTemperature)
     && guessedTemperature > 0.0
     && Foam::gpuThermal::finiteColdWallValue(ringSolidMass)
     && ringSolidMass >= 0.0
     && Foam::gpuThermal::finiteColdWallValue(profileContactAgeS)
     && profileContactAgeS >= 0.0
     && Foam::gpuThermal::finiteColdWallValue(frozenArea)
     && frozenArea >= 0.0;
    if (!__all_sync(mask, stateValid))
    {
        return false;
    }

    const double contactArea = Foam::gpuThermal::coldWallClamp
    (
        intrinsicContactAreaM2,
        DBL_MIN,
        maximumAreaM2
    );
    const double filmThickness = physicalVolumeM3/contactArea;
    const double nodeThickness = filmThickness/8.0;
    const double nodeMass = physicalMassKg/8.0;
    const double ringArea = maximumAreaM2/8.0;
    const double ringInnerArea = static_cast<double>(lane)*ringArea;
    const double ringWetArea = Foam::gpuThermal::coldWallClamp
    (
        contactArea - ringInnerArea,
        0.0,
        ringArea
    );
    const double radialFraction = (static_cast<double>(lane) + 0.5)/8.0;
    const double spreadCoordinate = 1.0 - ::sqrt
    (
        Foam::gpuThermal::coldWallClamp(1.0 - radialFraction, 0.0, 1.0)
    );
    const double firstWetAge =
        contactDurationS*peakTimeFraction*spreadCoordinate;
    const double ageMid = profileContactAgeS + 0.5*deltaTSeconds;
    const double localAge = ageMid > firstWetAge
      ? ageMid - firstWetAge
      : 0.0;
    double ringCoolingPower = 0.0;
    double acceptedWallPower = 0.0;

    for
    (
        int iteration = 0;
        iteration < s.coldWallSolidificationParameters.nonlinearIterations;
        ++iteration
    )
    {
        const double conductivity =
            Foam::gpuThermal::coldWallThermalConductivity
            (
                candidateEnthalpy,
                s.coldWallSolidificationParameters
            );
        const double bottomConductivity =
            __shfl_sync(mask, conductivity, 0, 8);
        const double particleResistance =
            0.5*nodeThickness/bottomConductivity;
        const double wallResistance =
            s.coldWallSolidificationParameters.wallTransientResistance != 0
          ? ::sqrt(Foam::gpuThermal::coldWallPi*localAge)/wallEffusivity
          : 0.0;
        const double resistance = particleResistance
          + s.coldWallSolidificationParameters.interfaceResistanceM2KW
          + wallResistance;
        const double ringConductance = ringWetArea > 0.0
          ? thermalAreaFactor*ringWetArea/(resistance + DBL_MIN)
          : 0.0;
        const double wallConductance =
            coldWall1DGroupSum(ringConductance, mask);
        const double bottomTemperature =
            __shfl_sync(mask, guessedTemperature, 0, 8);
        ringCoolingPower =
            ringConductance*(bottomTemperature - wallTemperatureK);

        const double rightConductivity =
            __shfl_down_sync(mask, conductivity, 1, 8);
        const double faceConductance = lane + 1 < 8
          ? 2.0*conductivity*rightConductivity
           /(conductivity + rightConductivity + DBL_MIN)
           *contactArea/nodeThickness
          : 0.0;
        const double leftFaceConductance =
            __shfl_up_sync(mask, faceConductance, 1, 8);
        const double capacity = nodeMass
          *Foam::gpuThermal::coldWallApparentSpecificHeat
            (
                guessedTemperature,
                s.coldWallSolidificationParameters
            )/(deltaTSeconds + DBL_MIN);
        double lower = lane > 0 ? -leftFaceConductance : 0.0;
        double diagonal = capacity
          + (lane > 0 ? leftFaceConductance : 0.0)
          + (lane + 1 < 8 ? faceConductance : 0.0);
        double upper = lane + 1 < 8 ? -faceConductance : 0.0;
        double rightHandSide = capacity*guessedTemperature;
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
        const double solvedTemperature = coldWall1DPcrSolve
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            lane,
            mask
        );
        const double rightSolvedTemperature =
            __shfl_down_sync(mask, solvedTemperature, 1, 8);
        const double internalPower = lane + 1 < 8
          ? faceConductance*(solvedTemperature - rightSolvedTemperature)
          : 0.0;
        const double leftInternalPower =
            __shfl_up_sync(mask, internalPower, 1, 8);
        const double wallPower = wallConductance
          *(__shfl_sync(mask, solvedTemperature, 0, 8) - wallTemperatureK);
        const double gasPower = gasConductanceWK
          *(gasTemperatureK - __shfl_sync(mask, solvedTemperature, 7, 8));
        double power =
            (lane > 0 ? leftInternalPower : 0.0)
          - (lane + 1 < 8 ? internalPower : 0.0);
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
         && candidateEnthalpy >= 0.0
         && Foam::gpuThermal::finiteColdWallValue(guessedTemperature)
         && guessedTemperature > 0.0;
        if (!__all_sync(mask, stateValid))
        {
            return false;
        }
    }

    double connectedFraction = Foam::gpuThermal::coldWallSolidFraction
    (
        candidateEnthalpy,
        s.coldWallSolidificationParameters
    );
    connectedFraction = coldWall1DGroupMinPrefix
    (
        connectedFraction,
        lane,
        mask
    );
    const double connectedMass = coldWall1DGroupSum
    (
        nodeMass*connectedFraction,
        mask
    );
    double assignedMass = coldWall1DGroupSum(ringSolidMass, mask);
    if (connectedMass < assignedMass && assignedMass > 0.0)
    {
        ringSolidMass *= connectedMass/assignedMass;
        assignedMass = connectedMass;
    }
    double remainingMass = connectedMass - assignedMass;
    const double ringCapacity =
        s.coldWallSolidificationParameters.solidDensityKgM3
       *ringWetArea*filmThickness;
    for
    (
        int pass = 0;
        pass < 8 && remainingMass > 1.0e-18*physicalMassKg;
        ++pass
    )
    {
        const double available = ringCapacity - ringSolidMass;
        const double cooling = ringCoolingPower > 0.0
          ? ringCoolingPower
          : 0.0;
        const double weight = available > 0.0
          ? (cooling > 0.0 ? cooling : ringWetArea)
          : 0.0;
        const double weightSum = coldWall1DGroupSum(weight, mask);
        if (!(weightSum > 0.0))
        {
            break;
        }
        const double requested = remainingMass*weight/weightSum;
        const double addition = available > 0.0
          ? (requested < available ? requested : available)
          : 0.0;
        ringSolidMass += addition;
        const double allocated = coldWall1DGroupSum(addition, mask);
        remainingMass -= allocated;
        if (!(allocated > 0.0))
        {
            break;
        }
    }

    const double localSolidFraction = ringWetArea > 0.0
      ? ringSolidMass
       /(s.coldWallSolidificationParameters.solidDensityKgM3
        *ringWetArea*filmThickness)
      : 0.0;
    const unsigned int eligibleMask = __ballot_sync
    (
        mask,
        ringWetArea > 0.0
     && localSolidFraction
        >= s.coldWallSolidificationParameters.pinningThicknessFraction
    ) >> (__ffs(mask) - 1);
    const unsigned int lowerMask = (1u << (lane + 1)) - 1u;
    const bool connectedRing =
        (eligibleMask & lowerMask) == lowerMask;
    const double candidateFrozenArea = coldWall1DGroupSum
    (
        connectedRing ? ringWetArea : 0.0,
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
            static_cast<float>(profileContactAgeS);
        s.pColdFrozenArea[particleI] = static_cast<float>(frozenArea);
    }
    meanTemperatureK =
        Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK
        (
            coldWall1DGroupSum(candidateEnthalpy, mask)/8.0,
            s.coldWallSolidificationParameters
        );
    frozenFootprintAreaM2 = frozenArea;
    wallEnergyJ = acceptedWallPower*deltaTSeconds;
    stateValid =
        Foam::gpuThermal::finiteColdWallValue(meanTemperatureK)
     && meanTemperatureK > 0.0
     && Foam::gpuThermal::finiteColdWallValue(frozenFootprintAreaM2)
     && frozenFootprintAreaM2 >= 0.0
     && Foam::gpuThermal::finiteColdWallValue(wallEnergyJ);
    return __all_sync(mask, stateValid);
}

#ifndef GPU_COLD_WALL_1D_ALGEBRA_ONLY

__global__ void relaxColdWall1DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const double dt
)
{
    DeviceState& s = *sp;
    if ((blockDim.x & 31) != 0)
    {
        asm("trap;");
    }
    const int lane = threadIdx.x & 7;
    const int groupsPerBlock = blockDim.x/8;
    const int groupInBlock = threadIdx.x/8;
    const int groupStride = gridDim.x*groupsPerBlock;
    const unsigned int mask = 0xffu << ((threadIdx.x & 31)/8*8);
    const int nWallBound = Foam::gpuWall::wallBoundDirectoryCount(s);
    for
    (
        int entry = blockIdx.x*groupsPerBlock + groupInBlock;
        entry < nWallBound;
        entry += groupStride
    )
    {
        const int i = Foam::gpuWall::wallBoundDirectoryParticle(s, entry);
        int active = i >= 0 && i < s.particleCapacity;
        active = active
          && s.pStatus[i] != 0
          && s.pStuck[i] != Foam::gpuThermal::particleWallMobile
          && s.particleWallHeatTransferEnabled != 0;
        const int faceI = active ? s.pStuckFaceId[i] : -1;
        active = active
          && faceI >= 0
          && faceI < s.nFaces
          && s.particleStuckCandidateMask[faceI]
             == Foam::gpuThermal::particleWallSolidifyingDeposition;
        if (!__all_sync(mask, active))
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            asm("trap;");
        }
        const double dPart = clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            1.0e-12
        );
        const double tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const Foam::gpuThermal::AluminaLiquidProperties material =
            Foam::gpuThermal::liquidAluminaProperties(tpOld);
        const double physicalVolume =
            (Foam::gpuThermal::finiteContactPi/6.0)*dPart*dPart*dPart;
        const double physicalMass = material.densityKgM3*physicalVolume;
        const double parcelMultiplicity = s.pm[i]/physicalMass;
        double gasConductanceWK = 0.0;
        const double gasTemperatureK = clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            1.0e30
        );
        if
        (
            s.solveParticleTemperature != 0
         && s.particleGasHeatTransferModelId != 0
        )
        {
            const double rhoG = clampMin
            (
                finiteOr(s.couplingRhoOld[c], s.rhoMin),
                s.rhoMin
            );
            const double ugx = finiteOr(s.couplingUxOld[c], 0.0);
            const double ugy = finiteOr(s.couplingUyOld[c], 0.0);
            const double ugz = finiteOr(s.couplingUzOld[c], 0.0);
            const double relMag = sqrt(ugx*ugx + ugy*ugy + ugz*ugz);
            const double re = rhoG*dPart*relMag/clampMin(s.gasMu, 1.0e-30);
            const double nu = 2.0
              + 0.6*sqrt(clampMin(re, 0.0))*s.gasPrOneThird;
            const double particleCp = particleSpecificHeatDevice(tpOld);
            const double rate = 6.0*nu*molecularGasConductivity(s)
              /(s.rhoSolid*particleCp*dPart*dPart + 1.0e-300);
            gasConductanceWK = rate*physicalMass*particleCp;
        }
        const unsigned char wallState = s.pStuck[i];
        const bool finiteContact =
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit;
        double maximumArea = static_cast<double>(s.pContactMaximumArea[i]);
        double intrinsicArea = 0.0;
        double duration = static_cast<double>(s.pContactDuration[i]);
        double peakTimeFraction =
            static_cast<double>(s.pContactPeakFraction[i]);
        double activeDt = dt;
        if (finiteContact)
        {
            const double damageArea =
                static_cast<double>(s.pDepositionArea[i]);
            const double age0 = clampRange
            (
                finiteOr(s.pTheta[i], 0.0),
                0.0,
                duration
            );
            activeDt = clampRange(dt, 0.0, duration - age0);
            const double ageMid = age0 + 0.5*activeDt;
            const double kinematicAreaMid = maximumArea
              *Foam::gpuThermal::normalizedKinematicArea
                (
                    ageMid/duration,
                    peakTimeFraction
                );
            intrinsicArea = clampMin
            (
                fmax
                (
                    kinematicAreaMid,
                    static_cast<double>(s.pColdFrozenArea[i])
                ) - damageArea,
                0.0
            );
        }
        else
        {
            intrinsicArea = static_cast<double>(s.pDepositionArea[i]);
        }
        if (!(activeDt > 0.0) || !(intrinsicArea > 0.0))
        {
            continue;
        }
        double meanTemperature = 0.0;
        double frozenArea = 0.0;
        double wallEnergy = 0.0;
        const double efficiency = finiteContact
          ? s.particleWallReflectionHeatTransferEfficiency
          : s.particleWallDepositionHeatTransferEfficiency;
        const double wallEffusivity =
            s.particleWallEffusivityByFace != nullptr
          ? s.particleWallEffusivityByFace[faceI]
          : sqrt
            (
                s.particleWallDensityKgM3
               *s.particleWallSpecificHeatJkgK
               *s.particleWallConductivityWmK
            );
        const bool valid = advanceColdWall1DThermalGroup
        (
            s,
            i,
            lane,
            mask,
            physicalVolume,
            physicalMass,
            maximumArea,
            intrinsicArea,
            duration,
            peakTimeFraction,
            activeDt,
            s.gasBoundaryT[faceI],
            wallEffusivity,
            efficiency*s.particleWallContactAreaScale[faceI],
            gasTemperatureK,
            gasConductanceWK,
            meanTemperature,
            frozenArea,
            wallEnergy
        );
        if (!valid || !(parcelMultiplicity > 0.0) || frozenArea > maximumArea)
        {
            asm("trap;");
        }
        if (lane == 0)
        {
            s.pT[i] = meanTemperature;
            atomicAddParticleWallEnergyByFace
            (
                s,
                finiteContact
                  ? s.particleWallReflectedEnergy
                  : s.particleWallDepositedEnergy,
                faceI,
                parcelMultiplicity*wallEnergy
            );
        }
    }
}

#endif

#endif
