#include "GpuPrecisionTypes.H"
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

constexpr GpuReal particleWallContactPi =
    GPU_R(3.141592653589793238462643383279502884);
constexpr GpuReal particleWallContactNormalSpeedFloorMps = GPU_R(1.0e-12);

struct ParticleWallContactResult
{
    bool valid;
    GpuReal particleTemperatureK;
    GpuReal particleEnergyChangeJ;
    GpuReal wallEnergyJ;
    GpuReal fraction;
    GpuReal strength;
    GpuReal physicalParticleMassKg;
    GpuReal weber;
    GpuReal ohnesorge;
    GpuReal xiMax;
    GpuTime contactDurationS;
    GpuReal integratedContactAreaM2S;
    GpuReal depositionAreaM2;
};

GPU_PARTICLE_WALL_CONTACT_HD inline bool finiteParticleWallContactValue
(
    const GpuReal value
) noexcept
{
    return value == value && value >= -GPU_REAL_MAX && value <= GPU_REAL_MAX;
}

GPU_PARTICLE_WALL_CONTACT_HD inline ParticleWallContactResult
invalidParticleWallContactResult() noexcept
{
    return
    {
        false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0),
        GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)
    };
}

GPU_PARTICLE_WALL_CONTACT_HD inline ParticleWallContactResult
lumpedParticleWallInterfaceContact
(
    const GpuReal particleTemperatureK,
    const GpuReal wallTemperatureK,
    const GpuReal physicalParticleMassKg,
    const GpuReal parcelMassKg,
    const GpuReal conductanceTimeIntegralJK
) noexcept
{
    if
    (
        !finiteParticleWallContactValue(particleTemperatureK)
     || !finiteParticleWallContactValue(wallTemperatureK)
     || !finiteParticleWallContactValue(physicalParticleMassKg)
     || !finiteParticleWallContactValue(parcelMassKg)
     || !finiteParticleWallContactValue(conductanceTimeIntegralJK)
     || !(particleTemperatureK > GPU_R(0.0))
     || !(wallTemperatureK > GPU_R(0.0))
     || !(physicalParticleMassKg > GPU_R(0.0))
     || !(parcelMassKg > GPU_R(0.0))
     || conductanceTimeIntegralJK < GPU_R(0.0)
    )
    {
        return invalidParticleWallContactResult();
    }
    if (!(conductanceTimeIntegralJK > GPU_R(0.0)))
    {
        return
        {
            true,
            particleTemperatureK,
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0),
            physicalParticleMassKg,
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0),
            GPU_R(0.0)
        };
    }

    const GpuReal oldSpecificEnthalpy =
        aluminaSpecificEnthalpyJkg(particleTemperatureK);
    GpuReal lower = particleTemperatureK < wallTemperatureK
      ? particleTemperatureK
      : wallTemperatureK;
    GpuReal upper = particleTemperatureK > wallTemperatureK
      ? particleTemperatureK
      : wallTemperatureK;
    for (int iteration = 0; iteration < 48; ++iteration)
    {
        const GpuReal middle = GPU_R(0.5)*(lower + upper);
        const GpuReal residual =
            physicalParticleMassKg
           *(aluminaSpecificEnthalpyJkg(middle) - oldSpecificEnthalpy)
          + conductanceTimeIntegralJK*(middle - wallTemperatureK);
        if (!finiteParticleWallContactValue(residual))
        {
            return invalidParticleWallContactResult();
        }
        if (residual < GPU_R(0.0))
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    const GpuReal newTemperatureK = GPU_R(0.5)*(lower + upper);
    const GpuReal physicalWallEnergyJ =
        physicalParticleMassKg
       *(
            oldSpecificEnthalpy
          - aluminaSpecificEnthalpyJkg(newTemperatureK)
        );
    const GpuReal parcelWallEnergyJ =
        physicalWallEnergyJ*parcelMassKg/physicalParticleMassKg;
    if
    (
        !finiteParticleWallContactValue(newTemperatureK)
     || !finiteParticleWallContactValue(physicalWallEnergyJ)
     || !finiteParticleWallContactValue(parcelWallEnergyJ)
     || !(newTemperatureK > GPU_R(0.0))
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
        GPU_R(0.0),
        conductanceTimeIntegralJK,
        physicalParticleMassKg,
        GPU_R(0.0),
        GPU_R(0.0),
        GPU_R(0.0),
        GPU_R(0.0),
        conductanceTimeIntegralJK,
        GPU_R(0.0)
    };
}

GPU_PARTICLE_WALL_CONTACT_HD inline GpuReal representedDepositionContactArea
(
    const GpuReal parcelReferenceDensityKgM3,
    const GpuReal diameterM,
    const GpuReal depositionAreaM2,
    const GpuReal parcelMassKg
) noexcept
{
    if
    (
        !finiteParticleWallContactValue(parcelReferenceDensityKgM3)
     || !finiteParticleWallContactValue(diameterM)
     || !finiteParticleWallContactValue(depositionAreaM2)
     || !finiteParticleWallContactValue(parcelMassKg)
     || !(parcelReferenceDensityKgM3 > GPU_R(0.0))
     || !(diameterM > GPU_R(0.0))
     || !(depositionAreaM2 > GPU_R(0.0))
     || !(parcelMassKg > GPU_R(0.0))
    )
    {
        return -GPU_R(1.0);
    }

    const GpuReal physicalParticleMassKg =
        (particleWallContactPi/GPU_R(6.0))*parcelReferenceDensityKgM3
       *diameterM*diameterM*diameterM;
    if
    (
        !finiteParticleWallContactValue(physicalParticleMassKg)
     || !(physicalParticleMassKg > GPU_R(0.0))
    )
    {
        return -GPU_R(1.0);
    }

    const GpuReal representedAreaM2 =
        depositionAreaM2*parcelMassKg/physicalParticleMassKg;
    return
        finiteParticleWallContactValue(representedAreaM2)
     && representedAreaM2 > GPU_R(0.0)
      ? representedAreaM2
      : -GPU_R(1.0);
}


}                        
}                  

#undef GPU_PARTICLE_WALL_CONTACT_HD

#endif

