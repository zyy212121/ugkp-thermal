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
    const double value
) noexcept
{
    const double maximum = 1.7976931348623157e+308;
    return value == value && value >= -maximum && value <= maximum;
}

GPU_PARTICLE_RADIATION_HD inline double invalidRadiationMathValue() noexcept
{
    return __builtin_nan("");
}

struct RadiationAffineTemperatureUpdate
{
    double scale;
    double offset;
};

GPU_PARTICLE_RADIATION_HD inline RadiationAffineTemperatureUpdate
radiationAffineTemperatureUpdate
(
    const double energyJ,
    const double particleMassKg,
    const double heatFactorJkgK,
    const double meanTemperatureK,
    const double minimumTemperatureK,
    const double maximumTemperatureK
) noexcept
{
    const double invalid = invalidRadiationMathValue();
    if
    (
        !finiteRadiationMathValue(energyJ)
     || !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || !finiteRadiationMathValue(meanTemperatureK)
     || !finiteRadiationMathValue(minimumTemperatureK)
     || !finiteRadiationMathValue(maximumTemperatureK)
     || particleMassKg < 0.0
     || !(heatFactorJkgK > 0.0)
     || !(minimumTemperatureK < maximumTemperatureK)
     || meanTemperatureK < minimumTemperatureK
     || meanTemperatureK > maximumTemperatureK
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    if (particleMassKg == 0.0)
    {
        return energyJ == 0.0
          ? RadiationAffineTemperatureUpdate{1.0, 0.0}
          : RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    if (energyJ == 0.0)
    {
        return RadiationAffineTemperatureUpdate{1.0, 0.0};
    }

    const double capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > 0.0)
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }

    const bool cooling = energyJ < 0.0;
    const double targetTemperatureK = cooling
      ? minimumTemperatureK
      : maximumTemperatureK;
    const double availableEnergyJ = cooling
      ? capacityJPerK*(meanTemperatureK - minimumTemperatureK)
      : capacityJPerK*(maximumTemperatureK - meanTemperatureK);
    const double requestedEnergyMagnitudeJ = cooling ? -energyJ : energyJ;
    if
    (
        !finiteRadiationMathValue(availableEnergyJ)
     || !(availableEnergyJ > 0.0)
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }

    const double fraction = requestedEnergyMagnitudeJ/availableEnergyJ;
    if
    (
        !finiteRadiationMathValue(fraction)
     || !(fraction > 0.0)
     || fraction > 1.0
    )
    {
        return RadiationAffineTemperatureUpdate{invalid, invalid};
    }
    const double scale = 1.0 - fraction;
    const double offsetK = fraction*targetTemperatureK;
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

GPU_PARTICLE_RADIATION_HD inline double applyRadiationAffineTemperature
(
    const double oldTemperatureK,
    const RadiationAffineTemperatureUpdate& update
) noexcept
{
    if
    (
        !finiteRadiationMathValue(oldTemperatureK)
     || !finiteRadiationMathValue(update.scale)
     || !finiteRadiationMathValue(update.offset)
     || update.scale < 0.0
     || update.scale > 1.0
    )
    {
        return invalidRadiationMathValue();
    }
    const double proposedTemperatureK =
        update.scale*oldTemperatureK + update.offset;
    return finiteRadiationMathValue(proposedTemperatureK)
      ? proposedTemperatureK
      : invalidRadiationMathValue();
}

                                                                               
                                                                                   
GPU_PARTICLE_RADIATION_HD inline double radiationDeltaTemperature
(
    const double energyJ,
    const double particleMassKg,
    const double heatFactorJkgK
) noexcept
{
    if
    (
        !finiteRadiationMathValue(energyJ)
     || !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || particleMassKg < 0.0
     || !(heatFactorJkgK > 0.0)
    )
    {
        return invalidRadiationMathValue();
    }

    if (particleMassKg == 0.0 && energyJ == 0.0)
    {
        return 0.0;
    }
    if (particleMassKg == 0.0)
    {
        return invalidRadiationMathValue();
    }

    const double capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > 0.0)
    )
    {
        return invalidRadiationMathValue();
    }

    const double deltaTemperatureK = energyJ/capacityJPerK;
    if
    (
        !finiteRadiationMathValue(deltaTemperatureK)
     || (energyJ != 0.0 && deltaTemperatureK == 0.0)
    )
    {
        return invalidRadiationMathValue();
    }
    return deltaTemperatureK;
}

                                                                               
GPU_PARTICLE_RADIATION_HD inline double radiationParticleEnergyChange
(
    const double particleMassKg,
    const double heatFactorJkgK,
    const double deltaTemperatureK
) noexcept
{
    if
    (
        !finiteRadiationMathValue(particleMassKg)
     || !finiteRadiationMathValue(heatFactorJkgK)
     || !finiteRadiationMathValue(deltaTemperatureK)
     || particleMassKg < 0.0
     || !(heatFactorJkgK > 0.0)
    )
    {
        return invalidRadiationMathValue();
    }
    if (particleMassKg == 0.0)
    {
        return 0.0;
    }

    const double capacityJPerK = particleMassKg*heatFactorJkgK;
    if
    (
        !finiteRadiationMathValue(capacityJPerK)
     || !(capacityJPerK > 0.0)
    )
    {
        return invalidRadiationMathValue();
    }

    const double energyJ = capacityJPerK*deltaTemperatureK;
    if
    (
        !finiteRadiationMathValue(energyJ)
     ||
        (
            particleMassKg != 0.0
         && deltaTemperatureK != 0.0
         && energyJ == 0.0
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
