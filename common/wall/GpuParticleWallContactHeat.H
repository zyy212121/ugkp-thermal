#ifndef GPU_THERMAL_GPU_PARTICLE_WALL_CONTACT_HEAT_H
#define GPU_THERMAL_GPU_PARTICLE_WALL_CONTACT_HEAT_H

#include "GpuAluminaProperties.H"
#include "GpuWallInterfaceHeat.H"

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define GPU_PARTICLE_WALL_CONTACT_HD __host__ __device__
#else
#define GPU_PARTICLE_WALL_CONTACT_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr double particleWallContactPi =
    3.141592653589793238462643383279502884;
constexpr double particleWallContactNormalSpeedFloorMps = 1.0e-12;

struct ParticleWallContactResult
{
    bool valid;
    double particleTemperatureK;
    double particleEnergyChangeJ;
    double wallEnergyJ;
    double fraction;
    double strength;
    double physicalParticleMassKg;
    double weber;
    double ohnesorge;
    double xiMax;
    double contactDurationS;
    double integratedContactAreaM2S;
    double depositionAreaM2;
};

GPU_PARTICLE_WALL_CONTACT_HD inline bool finiteParticleWallContactValue
(
    const double value
) noexcept
{
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

GPU_PARTICLE_WALL_CONTACT_HD inline ParticleWallContactResult
invalidParticleWallContactResult() noexcept
{
    return
    {
        false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
}

GPU_PARTICLE_WALL_CONTACT_HD inline ParticleWallContactResult
lumpedParticleWallInterfaceContact
(
    const double particleTemperatureK,
    const double wallTemperatureK,
    const double physicalParticleMassKg,
    const double parcelMassKg,
    const double conductanceTimeIntegralJK
) noexcept
{
    if
    (
        !finiteParticleWallContactValue(particleTemperatureK)
     || !finiteParticleWallContactValue(wallTemperatureK)
     || !finiteParticleWallContactValue(physicalParticleMassKg)
     || !finiteParticleWallContactValue(parcelMassKg)
     || !finiteParticleWallContactValue(conductanceTimeIntegralJK)
     || !(particleTemperatureK > 0.0)
     || !(wallTemperatureK > 0.0)
     || !(physicalParticleMassKg > 0.0)
     || !(parcelMassKg > 0.0)
     || conductanceTimeIntegralJK < 0.0
    )
    {
        return invalidParticleWallContactResult();
    }
    if (!(conductanceTimeIntegralJK > 0.0))
    {
        return
        {
            true,
            particleTemperatureK,
            0.0,
            0.0,
            0.0,
            0.0,
            physicalParticleMassKg,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        };
    }

    const double oldSpecificEnthalpy =
        aluminaSpecificEnthalpyJkg(particleTemperatureK);
    double lower = particleTemperatureK < wallTemperatureK
      ? particleTemperatureK
      : wallTemperatureK;
    double upper = particleTemperatureK > wallTemperatureK
      ? particleTemperatureK
      : wallTemperatureK;
    for (int iteration = 0; iteration < 48; ++iteration)
    {
        const double middle = 0.5*(lower + upper);
        const double residual =
            physicalParticleMassKg
           *(aluminaSpecificEnthalpyJkg(middle) - oldSpecificEnthalpy)
          + conductanceTimeIntegralJK*(middle - wallTemperatureK);
        if (!finiteParticleWallContactValue(residual))
        {
            return invalidParticleWallContactResult();
        }
        if (residual < 0.0)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    const double newTemperatureK = 0.5*(lower + upper);
    const double physicalWallEnergyJ =
        physicalParticleMassKg
       *(
            oldSpecificEnthalpy
          - aluminaSpecificEnthalpyJkg(newTemperatureK)
        );
    const double parcelWallEnergyJ =
        physicalWallEnergyJ*parcelMassKg/physicalParticleMassKg;
    if
    (
        !finiteParticleWallContactValue(newTemperatureK)
     || !finiteParticleWallContactValue(physicalWallEnergyJ)
     || !finiteParticleWallContactValue(parcelWallEnergyJ)
     || !(newTemperatureK > 0.0)
    )
    {
        return invalidParticleWallContactResult();
    }
    return
    {
        true,
        newTemperatureK,
        -parcelWallEnergyJ,
        parcelWallEnergyJ,
        0.0,
        conductanceTimeIntegralJK,
        physicalParticleMassKg,
        0.0,
        0.0,
        0.0,
        0.0,
        conductanceTimeIntegralJK,
        0.0
    };
}

GPU_PARTICLE_WALL_CONTACT_HD inline double representedDepositionContactArea
(
    const double parcelReferenceDensityKgM3,
    const double diameterM,
    const double depositionAreaM2,
    const double parcelMassKg
) noexcept
{
    if
    (
        !finiteParticleWallContactValue(parcelReferenceDensityKgM3)
     || !finiteParticleWallContactValue(diameterM)
     || !finiteParticleWallContactValue(depositionAreaM2)
     || !finiteParticleWallContactValue(parcelMassKg)
     || !(parcelReferenceDensityKgM3 > 0.0)
     || !(diameterM > 0.0)
     || !(depositionAreaM2 > 0.0)
     || !(parcelMassKg > 0.0)
    )
    {
        return -1.0;
    }

    const double physicalParticleMassKg =
        (particleWallContactPi/6.0)*parcelReferenceDensityKgM3
       *diameterM*diameterM*diameterM;
    if
    (
        !finiteParticleWallContactValue(physicalParticleMassKg)
     || !(physicalParticleMassKg > 0.0)
    )
    {
        return -1.0;
    }

    const double representedAreaM2 =
        depositionAreaM2*parcelMassKg/physicalParticleMassKg;
    return
        finiteParticleWallContactValue(representedAreaM2)
     && representedAreaM2 > 0.0
      ? representedAreaM2
      : -1.0;
}


}                        
}                  

#undef GPU_PARTICLE_WALL_CONTACT_HD

#endif

