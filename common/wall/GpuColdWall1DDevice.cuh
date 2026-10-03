#include "GpuPrecisionTypes.H"
#include "GpuColdWall1DAlgebra.cuh"
#if UGKWP_GPU_REAL_BITS == 32
#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH
#define GPU_THERMAL_GPU_COLD_WALL_1D_DEVICE_CUH












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
#include "GpuColdWall1DAdvance.inl"
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
#include "GpuColdWall1DAdvance.inl"
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
