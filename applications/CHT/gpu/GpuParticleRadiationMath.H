#include "GpuPrecisionTypes.H"
#ifndef GPU_PARTICLE_RADIATION_MATH_H
#define GPU_PARTICLE_RADIATION_MATH_H

#include <limits>

#if defined(__CUDACC__)
#define GPU_PARTICLE_RADIATION_HD __host__ __device__
#else
#define GPU_PARTICLE_RADIATION_HD
#endif

namespace Foam
{
namespace gpuThermal
{

GPU_PARTICLE_RADIATION_HD inline bool finiteRadiationMathValue
(
    const GpuReal value
) noexcept
{
    const GpuReal maximum = GPU_LARGE(1.7976931348623157e+308);
    return value == value && value >= -maximum && value <= maximum;
}

GPU_PARTICLE_RADIATION_HD inline GpuReal invalidRadiationMathValue() noexcept
{
    return __builtin_nan("");
}

struct RadiationAffineTemperatureUpdate
{
    GpuReal scale;
    GpuReal offset;
};

GPU_PARTICLE_RADIATION_HD inline RadiationAffineTemperatureUpdate
radiationAffineTemperatureUpdate
(
    const GpuReal energyJ,
    const GpuReal particleMassKg,
    const GpuReal heatFactorJkgK,
    const GpuReal meanTemperatureK,
    const GpuReal minimumTemperatureK,
    const GpuReal maximumTemperatureK
) noexcept
{
    const GpuReal invalid = invalidRadiationMathValue();
    if
    (
        !finiteRadiationMathValue(energyJ)
     || !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || !finiteRadiationMathValue(meanTemperatureK)
     || !finiteRadiationMathValue(minimumTemperatureK)
     || !finiteRadiationMathValue(maximumTemperatureK)
     || particleMassKg < GPU_R(0.0)
     || !(heatFactorJkgK > GPU_R(0.0))
     || !(minimumTemperatureK < maximumTemperatureK)
     || meanTemperatureK < minimumTemperatureK
     || meanTemperatureK > maximumTemperatureK
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    if (particleMassKg == GPU_R(0.0))
    {
        return energyJ == GPU_R(0.0)
          ? RadiationAffineTemperatureUpdate{GPU_R(1.0), GPU_R(0.0)}
          : RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    if (energyJ == GPU_R(0.0))
    {
        return RadiationAffineTemperatureUpdate{GPU_R(1.0), GPU_R(0.0)};
    }

    const GpuReal capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > GPU_R(0.0))
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }

    const bool cooling = energyJ < GPU_R(0.0);
    const GpuReal targetTemperatureK = cooling
      ? minimumTemperatureK
      : maximumTemperatureK;
    const GpuReal availableEnergyJ = cooling
      ? capacityJPerK*(meanTemperatureK - minimumTemperatureK)
      : capacityJPerK*(maximumTemperatureK - meanTemperatureK);
    const GpuReal requestedEnergyMagnitudeJ = cooling ? -energyJ : energyJ;
    if
    (
        !finiteRadiationMathValue(availableEnergyJ)
     || !(availableEnergyJ > GPU_R(0.0))
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }

    const GpuReal fraction = requestedEnergyMagnitudeJ/availableEnergyJ;
    if
    (
        !finiteRadiationMathValue(fraction)
     || !(fraction > GPU_R(0.0))
     || fraction > GPU_R(1.0)
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    const GpuReal scale = GPU_R(1.0) - fraction;
    const GpuReal offsetK = fraction*targetTemperatureK;
    if
    (
        !finiteRadiationMathValue(scale)
     || !finiteRadiationMathValue(offsetK)
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    return RadiationAffineTemperatureUpdate{scale, offsetK};
}

GPU_PARTICLE_RADIATION_HD inline GpuReal applyRadiationAffineTemperature
(
    const GpuReal oldTemperatureK,
    const RadiationAffineTemperatureUpdate& update
) noexcept
{
    if
    (
        !finiteRadiationMathValue(oldTemperatureK)
     || !finiteRadiationMathValue(update.scale)
     || !finiteRadiationMathValue(update.offset)
     || update.scale < GPU_R(0.0)
     || update.scale > GPU_R(1.0)
    )
    {
        return invalidRadiationMathValue();
    }
    const GpuReal proposedTemperatureK =
        update.scale*oldTemperatureK + update.offset;
    return finiteRadiationMathValue(proposedTemperatureK)
      ? proposedTemperatureK
      : invalidRadiationMathValue();
}

                                                                               
                                                                                   
GPU_PARTICLE_RADIATION_HD inline GpuReal radiationDeltaTemperature
(
    const GpuReal energyJ,
    const GpuReal particleMassKg,
    const GpuReal heatFactorJkgK
) noexcept
{
    if
    (
        !finiteRadiationMathValue(energyJ)
     || !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || particleMassKg < GPU_R(0.0)
     || !(heatFactorJkgK > GPU_R(0.0))
    )
    {
        return invalidRadiationMathValue();
    }

    if (particleMassKg == GPU_R(0.0) && energyJ == GPU_R(0.0))
    {
        return GPU_R(0.0);
    }
    if (particleMassKg == GPU_R(0.0))
    {
        return invalidRadiationMathValue();
    }

    const GpuReal capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > GPU_R(0.0))
    )
    {
        return invalidRadiationMathValue();
    }

    const GpuReal deltaTemperatureK = energyJ/capacityJPerK;
    if
    (
        !finiteRadiationMathValue(deltaTemperatureK)
     || (energyJ != GPU_R(0.0) && deltaTemperatureK == GPU_R(0.0))
    )
    {
        return invalidRadiationMathValue();
    }
    return deltaTemperatureK;
}

                                                                               
GPU_PARTICLE_RADIATION_HD inline GpuReal radiationParticleEnergyChange
(
    const GpuReal particleMassKg,
    const GpuReal heatFactorJkgK,
    const GpuReal deltaTemperatureK
) noexcept
{
    if
    (
        !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || !finiteRadiationMathValue(deltaTemperatureK)
     || particleMassKg < GPU_R(0.0)
     || !(heatFactorJkgK > GPU_R(0.0))
    )
    {
        return invalidRadiationMathValue();
    }
    if (particleMassKg == GPU_R(0.0))
    {
        return GPU_R(0.0);
    }

    const GpuReal capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > GPU_R(0.0))
    )
    {
        return invalidRadiationMathValue();
    }

    const GpuReal energyJ = capacityJPerK*deltaTemperatureK;
    if
    (
        !finiteRadiationMathValue(energyJ)
     ||
        (
            particleMassKg != GPU_R(0.0)
         && deltaTemperatureK != GPU_R(0.0)
         && energyJ == GPU_R(0.0)
        )
    )
    {
        return invalidRadiationMathValue();
    }
    return energyJ;
}

}                        
}                  

#undef GPU_PARTICLE_RADIATION_HD

#endif
