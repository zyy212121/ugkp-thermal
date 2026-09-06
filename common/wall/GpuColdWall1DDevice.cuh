#if UGKWP_GPU_REAL_BITS == 32
#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH

__device__ inline GpuReal coldWall1DGroupSum
(
    GpuReal value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

__device__ inline GpuReal coldWall1DGroupMinPrefix
(
    GpuReal value,
    const int lane,
    const unsigned int mask
)
{
    for (int offset = 1; offset < 8; offset *= 2)
    {
        const GpuReal lower = __shfl_up_sync(mask, value, offset, 8);
        if (lane >= offset && lower < value)
        {
            value = lower;
        }
    }
    return value;
}

__device__ inline GpuReal coldWall1DPcrSolve
(
    GpuReal lower,
    GpuReal diagonal,
    GpuReal upper,
    GpuReal rightHandSide,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const GpuReal lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const GpuReal lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const GpuReal lowerUpper =
            __shfl_up_sync(mask, upper, stride, 8);
        const GpuReal lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const GpuReal upperLower =
            __shfl_down_sync(mask, lower, stride, 8);
        const GpuReal upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const GpuReal upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const GpuReal upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const GpuReal alpha = lane >= stride
          ? -lower/(lowerDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal beta = lane + stride < 8
          ? -upper/(upperDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal nextLower = lane >= stride
          ? alpha*lowerLower
          : GPU_R(0.0);
        const GpuReal nextUpper = lane + stride < 8
          ? beta*upperUpper
          : GPU_R(0.0);
        const GpuReal nextDiagonal = diagonal
          + alpha*lowerUpper + beta*upperLower;
        const GpuReal nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + GPU_REAL_MIN);
}






__device__ inline GpuReal coldWall1DPcrSolveStable
(
    GpuReal lower,
    GpuReal diagonal,
    GpuReal upper,
    GpuReal rightHandSide,
    GpuReal rowSum,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const GpuReal lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const GpuReal lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const GpuReal lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const GpuReal upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const GpuReal upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const GpuReal upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const GpuReal lowerRowSum = __shfl_up_sync(mask, rowSum, stride, 8);
        const GpuReal upperRowSum = __shfl_down_sync(mask, rowSum, stride, 8);
        const GpuReal alpha = lane >= stride
          ? -lower/(lowerDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal beta = lane + stride < 8
          ? -upper/(upperDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal nextLower = lane >= stride
          ? alpha*lowerLower
          : GPU_R(0.0);
        const GpuReal nextUpper = lane + stride < 8
          ? beta*upperUpper
          : GPU_R(0.0);
        const GpuReal nextRowSum = rowSum + alpha*lowerRowSum + beta*upperRowSum;
        const GpuReal nextDiagonal = nextRowSum - nextLower - nextUpper;
        const GpuReal nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        rowSum = nextRowSum;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + GPU_REAL_MIN);
}

__device__ inline GpuReal coldWall1DSolidFractionFromKnownTemperature
(
    const GpuReal temperatureK,
    const Foam::gpuThermal::ColdWallSolidificationParameters& parameters
)
{
    const GpuReal solidus = Foam::gpuThermal::coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = Foam::gpuThermal::coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus) return GPU_R(1.0);
    if (temperatureK >= liquidus) return GPU_R(0.0);
    return (liquidus - temperatureK)/parameters.mushyRangeK;
}



struct ColdWallIntegralCache
{
    GpuTime deltaAge=0, root0=0, rootDifference=0;
    GpuReal c=0;
};
__device__ inline ColdWallIntegralCache coldWallPrepareIntegralCache
(
    GpuTime age0S, GpuTime age1S, GpuReal wallEffusivityWsHalfM2K,
    bool wallTransientResistance
)
{
    using namespace Foam::gpuThermal;
    ColdWallIntegralCache cache;
    cache.deltaAge = age1S - age0S;
    if (wallTransientResistance && cache.deltaAge > GPU_R(0.0)
        && age0S >= GPU_R(0.0) && age1S >= age0S
        && wallEffusivityWsHalfM2K > GPU_R(0.0))
    {
        cache.c = ::sqrt(wallInterfacePi)/wallEffusivityWsHalfM2K;
        cache.root0 = ::sqrt(age0S);
        const GpuTime root1 = ::sqrt(age1S);
#if UGKWP_GPU_REAL_BITS == 32
        cache.rootDifference = cache.deltaAge/(root1 + cache.root0);
#else
        cache.rootDifference = root1 - cache.root0;
#endif
    }
    return cache;
}
__device__ inline GpuReal coldWallCachedResistanceTimeIntegral
(
    const ColdWallIntegralCache& cache,
    const GpuTime age0S,
    const GpuTime age1S,
    const GpuReal baseResistanceM2KW,
    const GpuReal wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    using namespace Foam::gpuThermal;
    if
    (
        !finiteWallInterfaceValue(age0S)
     || !finiteWallInterfaceValue(age1S)
     || !finiteWallInterfaceValue(baseResistanceM2KW)
     || !finiteWallInterfaceValue(wallEffusivityWsHalfM2K)
     || age0S < GPU_R(0.0)
     || age1S < age0S
     || baseResistanceM2KW < GPU_R(0.0)
     || !(wallEffusivityWsHalfM2K > GPU_R(0.0))
    )
    {
        return -GPU_R(1.0);
    }
    const GpuTime deltaAge = cache.deltaAge;
    if (!(deltaAge > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    if (!wallTransientResistance)
    {
        return baseResistanceM2KW > GPU_R(0.0)
          ? deltaAge/baseResistanceM2KW
          : -GPU_R(1.0);
    }
    const GpuReal c = cache.c;
    const GpuTime root0 = cache.root0;
    const GpuTime rootDifference = cache.rootDifference;
    if (!(c > GPU_R(0.0)) || !finiteWallInterfaceValue(c))
    {
        return -GPU_R(1.0);
    }
    if (!(baseResistanceM2KW > GPU_R(0.0)))
    {
        const GpuReal integral = GPU_R(2.0)*rootDifference/c;
        return finiteWallInterfaceValue(integral) && integral >= GPU_R(0.0)
          ? integral
          : -GPU_R(1.0);
    }
    const GpuReal x0 = c*root0;
    const GpuReal dx = c*rootDifference;
    const GpuReal denominator0 = baseResistanceM2KW + x0;
    const GpuReal ratioIncrement = dx/denominator0;
    const GpuReal bracket =
        x0*ratioIncrement
      - baseResistanceM2KW*wallInterfaceLog1pMinusX(ratioIncrement);
    const GpuReal integral = GPU_R(2.0)*bracket/(c*c);
    return finiteWallInterfaceValue(integral) && integral >= GPU_R(0.0)
      ? integral
      : -GPU_R(1.0);
}

__device__ inline GpuReal coldWallCachedConductanceTimeIntegral
(
    const ColdWallIntegralCache& cache,
    const GpuReal areaM2,
    const GpuReal efficiency,
    const GpuTime age0S,
    const GpuTime age1S,
    const GpuReal interfaceResistanceM2KW,
    const GpuReal particleAdditionalResistanceM2KW,
    const GpuReal wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    using namespace Foam::gpuThermal;
    if
    (
        !finiteWallInterfaceValue(areaM2)
     || !finiteWallInterfaceValue(efficiency)
     || !finiteWallInterfaceValue(interfaceResistanceM2KW)
     || !finiteWallInterfaceValue(particleAdditionalResistanceM2KW)
     || !(areaM2 >= GPU_R(0.0))
     || !(efficiency > GPU_R(0.0))
     || efficiency > GPU_R(1.0)
     || interfaceResistanceM2KW < GPU_R(0.0)
     || particleAdditionalResistanceM2KW < GPU_R(0.0)
    )
    {
        return -GPU_R(1.0);
    }
    if (!(areaM2 > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    const GpuReal resistanceIntegral = coldWallCachedResistanceTimeIntegral
    (
        cache,
        age0S,
        age1S,
        interfaceResistanceM2KW + particleAdditionalResistanceM2KW,
        wallEffusivityWsHalfM2K,
        wallTransientResistance
    );
    const GpuReal conductanceIntegral =
        efficiency*areaM2*resistanceIntegral;
    return
        resistanceIntegral >= GPU_R(0.0)
     && finiteWallInterfaceValue(conductanceIntegral)
     && conductanceIntegral >= GPU_R(0.0)
      ? conductanceIntegral
      : -GPU_R(1.0);
}


template<bool StabilizePcr = false>
__device__ bool advanceColdWall1DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int lane,
    const unsigned int mask,
    const GpuReal physicalVolumeM3,
    const GpuReal physicalMassKg,
    const GpuReal maximumAreaM2,
    const GpuReal intrinsicContactAreaM2,
    const GpuTime contactDurationS,
    const GpuReal peakTimeFraction,
    const GpuTime deltaTSeconds,
    const GpuReal wallTemperatureK,
    const GpuReal wallEffusivity,
    const GpuReal thermalAreaFactor,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK,
    GpuReal& meanTemperatureK,
    GpuReal& frozenFootprintAreaM2,
    GpuReal& wallEnergyJ,
    bool* linearSolveFailed = nullptr
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
    const ColdWallIntegralCache integralCache = coldWallPrepareIntegralCache
    (
        localAge0, localAge1, wallEffusivity,
        s.coldWallSolidificationParameters.wallTransientResistance != 0
    );
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
            if (linearSolveFailed != nullptr) *linearSolveFailed = true;
            return false;
        }
    }

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
}

#ifndef GPU_COLD_WALL_1D_ALGEBRA_ONLY

__global__ __launch_bounds__(256, 6) void relaxColdWall1DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const GpuTime dt
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
        const GpuReal dPart = clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_R(1.0e-12)
        );
        const GpuReal tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const Foam::gpuThermal::AluminaLiquidProperties material =
            Foam::gpuThermal::liquidAluminaProperties(tpOld);
        const GpuReal physicalVolume =
            (Foam::gpuThermal::finiteContactPi/GPU_R(6.0))*dPart*dPart*dPart;
        const GpuReal physicalMass = material.densityKgM3*physicalVolume;
        const GpuReal parcelMultiplicity = s.pm[i]/physicalMass;
        GpuReal gasConductanceWK = GPU_R(0.0);
        const GpuReal gasTemperatureK = clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            GPU_R(1.0e30)
        );
        if
        (
            s.solveParticleTemperature != 0
         && s.particleGasHeatTransferModelId != 0
        )
        {
            const GpuReal rhoG = clampMin
            (
                finiteOr(s.couplingRhoOld[c], s.rhoMin),
                s.rhoMin
            );
            const GpuReal ugx = finiteOr(s.couplingUxOld[c], GPU_R(0.0));
            const GpuReal ugy = finiteOr(s.couplingUyOld[c], GPU_R(0.0));
            const GpuReal ugz = finiteOr(s.couplingUzOld[c], GPU_R(0.0));
            const GpuReal relMag = sqrt(ugx*ugx + ugy*ugy + ugz*ugz);
            const GpuReal re = rhoG*dPart*relMag/clampMin(s.gasMu, GPU_R(1.0e-30));
            const GpuReal nu = GPU_R(2.0)
              + GPU_R(0.6)*sqrt(clampMin(re, GPU_R(0.0)))*s.gasPrOneThird;
            const GpuReal particleCp = particleSpecificHeatDevice(tpOld);
            const GpuReal rate = GPU_R(6.0)*nu*molecularGasConductivity(s)
              /(s.rhoSolid*particleCp*dPart*dPart + GPU_TINY(1.0e-300));
            gasConductanceWK = rate*physicalMass*particleCp;
        }
        const unsigned char wallState = s.pStuck[i];
        const bool finiteContact =
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit;
        GpuReal maximumArea = static_cast<GpuReal>(s.pContactMaximumArea[i]);
        GpuReal intrinsicArea = GPU_R(0.0);
        GpuTime duration = static_cast<GpuTime>(s.pContactDuration[i]);
        GpuReal peakTimeFraction =
            static_cast<GpuReal>(s.pContactPeakFraction[i]);
        GpuTime activeDt = dt;
        if (finiteContact)
        {
            const GpuReal damageArea =
                static_cast<GpuReal>(s.pDepositionArea[i]);
            const GpuTime age0 = clampRange
            (
                finiteOr(s.pContactAge[i], GPU_R(0.0)),
                GPU_R(0.0),
                duration
            );
            activeDt = clampRange(dt, GPU_R(0.0), duration - age0);
            const GpuTime ageMid = age0 + GPU_R(0.5)*activeDt;
            const GpuReal kinematicAreaMid = maximumArea
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
                    static_cast<GpuReal>(s.pColdFrozenArea[i])
                ) - damageArea,
                GPU_R(0.0)
            );
        }
        else
        {
            intrinsicArea = static_cast<GpuReal>(s.pDepositionArea[i]);
        }
        if (!(activeDt > GPU_R(0.0)) || !(intrinsicArea > GPU_R(0.0)))
        {
            continue;
        }
        GpuReal meanTemperature = GPU_R(0.0);
        GpuReal frozenArea = GPU_R(0.0);
        GpuReal wallEnergy = GPU_R(0.0);
        const GpuReal efficiency = finiteContact
          ? s.particleWallReflectionHeatTransferEfficiency
          : s.particleWallDepositionHeatTransferEfficiency;
        const GpuReal wallEffusivity =
            s.particleWallEffusivityByFace != nullptr
          ? s.particleWallEffusivityByFace[faceI]
          : sqrt
            (
                s.particleWallDensityKgM3
               *s.particleWallSpecificHeatJkgK
               *s.particleWallConductivityWmK
            );
        bool linearSolveFailed = false;
        bool valid = advanceColdWall1DThermalGroup
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
            wallEnergy,
            &linearSolveFailed
        );
        if (!valid && linearSolveFailed)
        {
        valid = advanceColdWall1DThermalGroup<true>
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
        }
        if (!valid || !(parcelMultiplicity > GPU_R(0.0)) || frozenArea > maximumArea)
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

#else
#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH

__device__ inline GpuReal coldWall1DGroupSum
(
    GpuReal value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

__device__ inline GpuReal coldWall1DGroupMinPrefix
(
    GpuReal value,
    const int lane,
    const unsigned int mask
)
{
    for (int offset = 1; offset < 8; offset *= 2)
    {
        const GpuReal lower = __shfl_up_sync(mask, value, offset, 8);
        if (lane >= offset && lower < value)
        {
            value = lower;
        }
    }
    return value;
}

__device__ inline GpuReal coldWall1DPcrSolve
(
    GpuReal lower,
    GpuReal diagonal,
    GpuReal upper,
    GpuReal rightHandSide,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const GpuReal lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const GpuReal lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const GpuReal lowerUpper =
            __shfl_up_sync(mask, upper, stride, 8);
        const GpuReal lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const GpuReal upperLower =
            __shfl_down_sync(mask, lower, stride, 8);
        const GpuReal upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const GpuReal upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const GpuReal upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const GpuReal alpha = lane >= stride
          ? -lower/(lowerDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal beta = lane + stride < 8
          ? -upper/(upperDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal nextLower = lane >= stride
          ? alpha*lowerLower
          : GPU_R(0.0);
        const GpuReal nextUpper = lane + stride < 8
          ? beta*upperUpper
          : GPU_R(0.0);
        const GpuReal nextDiagonal = diagonal
          + alpha*lowerUpper + beta*upperLower;
        const GpuReal nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + GPU_REAL_MIN);
}

__device__ inline GpuReal coldWall1DSolidFractionFromKnownTemperature
(
    const GpuReal temperatureK,
    const Foam::gpuThermal::ColdWallSolidificationParameters& parameters
)
{
    const GpuReal solidus = Foam::gpuThermal::coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = Foam::gpuThermal::coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus) return GPU_R(1.0);
    if (temperatureK >= liquidus) return GPU_R(0.0);
    return (liquidus - temperatureK)/parameters.mushyRangeK;
}
__device__ bool advanceColdWall1DThermalGroup
(
    DeviceState& s,
    const int particleI,
    const int lane,
    const unsigned int mask,
    const GpuReal physicalVolumeM3,
    const GpuReal physicalMassKg,
    const GpuReal maximumAreaM2,
    const GpuReal intrinsicContactAreaM2,
    const GpuTime contactDurationS,
    const GpuReal peakTimeFraction,
    const GpuTime deltaTSeconds,
    const GpuReal wallTemperatureK,
    const GpuReal wallEffusivity,
    const GpuReal thermalAreaFactor,
    const GpuReal gasTemperatureK,
    const GpuReal gasConductanceWK,
    GpuReal& meanTemperatureK,
    GpuReal& frozenFootprintAreaM2,
    GpuReal& wallEnergyJ
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
        const GpuReal solvedTemperature = coldWall1DPcrSolve
        (
            lower,
            diagonal,
            upper,
            rightHandSide,
            lane,
            mask
        );
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
            return false;
        }
    }

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
}

#ifndef GPU_COLD_WALL_1D_ALGEBRA_ONLY

__global__ void relaxColdWall1DParticlesToResidentGasKernel
(
    DeviceState* sp,
    const GpuTime dt
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
        const GpuReal dPart = clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_R(1.0e-12)
        );
        const GpuReal tpOld = clampRange(s.pT[i], s.TpMin, s.TpMax);
        const Foam::gpuThermal::AluminaLiquidProperties material =
            Foam::gpuThermal::liquidAluminaProperties(tpOld);
        const GpuReal physicalVolume =
            (Foam::gpuThermal::finiteContactPi/GPU_R(6.0))*dPart*dPart*dPart;
        const GpuReal physicalMass = material.densityKgM3*physicalVolume;
        const GpuReal parcelMultiplicity = s.pm[i]/physicalMass;
        GpuReal gasConductanceWK = GPU_R(0.0);
        const GpuReal gasTemperatureK = clampRange
        (
            finiteOr(s.couplingTgasOld[c], s.TgasMin),
            s.TgasMin,
            GPU_R(1.0e30)
        );
        if
        (
            s.solveParticleTemperature != 0
         && s.particleGasHeatTransferModelId != 0
        )
        {
            const GpuReal rhoG = clampMin
            (
                finiteOr(s.couplingRhoOld[c], s.rhoMin),
                s.rhoMin
            );
            const GpuReal ugx = finiteOr(s.couplingUxOld[c], GPU_R(0.0));
            const GpuReal ugy = finiteOr(s.couplingUyOld[c], GPU_R(0.0));
            const GpuReal ugz = finiteOr(s.couplingUzOld[c], GPU_R(0.0));
            const GpuReal relMag = sqrt(ugx*ugx + ugy*ugy + ugz*ugz);
            const GpuReal re = rhoG*dPart*relMag/clampMin(s.gasMu, GPU_R(1.0e-30));
            const GpuReal nu = GPU_R(2.0)
              + GPU_R(0.6)*sqrt(clampMin(re, GPU_R(0.0)))*s.gasPrOneThird;
            const GpuReal particleCp = particleSpecificHeatDevice(tpOld);
            const GpuReal rate = GPU_R(6.0)*nu*molecularGasConductivity(s)
              /(s.rhoSolid*particleCp*dPart*dPart + GPU_TINY(1.0e-300));
            gasConductanceWK = rate*physicalMass*particleCp;
        }
        const unsigned char wallState = s.pStuck[i];
        const bool finiteContact =
            wallState == Foam::gpuThermal::particleWallTransientRebound
         || wallState == Foam::gpuThermal::particleWallTransientDeposit;
        GpuReal maximumArea = static_cast<GpuReal>(s.pContactMaximumArea[i]);
        GpuReal intrinsicArea = GPU_R(0.0);
        GpuTime duration = static_cast<GpuTime>(s.pContactDuration[i]);
        GpuReal peakTimeFraction =
            static_cast<GpuReal>(s.pContactPeakFraction[i]);
        GpuTime activeDt = dt;
        if (finiteContact)
        {
            const GpuReal damageArea =
                static_cast<GpuReal>(s.pDepositionArea[i]);
            const GpuTime age0 = clampRange
            (
                finiteOr(s.pContactAge[i], GPU_R(0.0)),
                GPU_R(0.0),
                duration
            );
            activeDt = clampRange(dt, GPU_R(0.0), duration - age0);
            const GpuTime ageMid = age0 + GPU_R(0.5)*activeDt;
            const GpuReal kinematicAreaMid = maximumArea
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
                    static_cast<GpuReal>(s.pColdFrozenArea[i])
                ) - damageArea,
                GPU_R(0.0)
            );
        }
        else
        {
            intrinsicArea = static_cast<GpuReal>(s.pDepositionArea[i]);
        }
        if (!(activeDt > GPU_R(0.0)) || !(intrinsicArea > GPU_R(0.0)))
        {
            continue;
        }
        GpuReal meanTemperature = GPU_R(0.0);
        GpuReal frozenArea = GPU_R(0.0);
        GpuReal wallEnergy = GPU_R(0.0);
        const GpuReal efficiency = finiteContact
          ? s.particleWallReflectionHeatTransferEfficiency
          : s.particleWallDepositionHeatTransferEfficiency;
        const GpuReal wallEffusivity =
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
        if (!valid || !(parcelMultiplicity > GPU_R(0.0)) || frozenArea > maximumArea)
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

#endif
