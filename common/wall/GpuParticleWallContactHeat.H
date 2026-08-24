#ifndef GPU_THERMAL_GPU_PARTICLE_WALL_CONTACT_HEAT_H
#define GPU_THERMAL_GPU_PARTICLE_WALL_CONTACT_HEAT_H

#include "GpuAluminaProperties.H"

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

GPU_PARTICLE_WALL_CONTACT_HD inline double boundedContactFraction
(
    const double strength
) noexcept
{
    if (!(strength > 0.0))
    {
        return 0.0;
    }
    if (!finiteParticleWallContactValue(strength) || strength > 745.0)
    {
        return 1.0;
    }
    const double fraction = -::expm1(-strength);
    return fraction < 0.0 ? 0.0 : (fraction > 1.0 ? 1.0 : fraction);
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

GPU_PARTICLE_WALL_CONTACT_HD inline ParticleWallContactResult
depositedParticleWallContact
(
    const double diameterM,
    const double depositionAreaM2,
    const double depositionHeatTransferEfficiency,
    const double contactAreaScale,
    const double deltaTSeconds,
    const double particleTemperatureK,
    const double wallTemperatureK,
    const double parcelMassKg
) noexcept
{
    if
    (
        !finiteParticleWallContactValue(diameterM)
     || !finiteParticleWallContactValue(depositionAreaM2)
     || !finiteParticleWallContactValue(depositionHeatTransferEfficiency)
     || !finiteParticleWallContactValue(contactAreaScale)
     || !finiteParticleWallContactValue(deltaTSeconds)
     || !finiteParticleWallContactValue(particleTemperatureK)
     || !finiteParticleWallContactValue(wallTemperatureK)
     || !finiteParticleWallContactValue(parcelMassKg)
     || !(diameterM > 0.0)
     || !(depositionAreaM2 > 0.0)
     || !(depositionHeatTransferEfficiency > 0.0)
     || depositionHeatTransferEfficiency > 1.0
     || !(contactAreaScale > 0.0)
     || contactAreaScale > 1.0
     || deltaTSeconds < 0.0
     || !(particleTemperatureK > 0.0)
     || !(wallTemperatureK > 0.0)
     || !(parcelMassKg > 0.0)
    )
    {
        return invalidParticleWallContactResult();
    }

    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if (!material.valid)
    {
        return invalidParticleWallContactResult();
    }

    const double physicalVolume =
        (particleWallContactPi/6.0)*diameterM*diameterM*diameterM;
    const double physicalMass = material.densityKgM3*physicalVolume;
                                                                            
                                                                       
                                                                         
                         
    const double length = physicalVolume/depositionAreaM2;
    const double alpha =
        material.thermalConductivityWmK
       /(material.densityKgM3*material.specificHeatJkgK);
    double strength = alpha*deltaTSeconds/(length*length);
    if
    (
        !finiteParticleWallContactValue(physicalVolume)
     || !finiteParticleWallContactValue(physicalMass)
     || !finiteParticleWallContactValue(length)
     || !finiteParticleWallContactValue(alpha)
     || !(physicalVolume > 0.0)
     || !(physicalMass > 0.0)
     || !(length > 0.0)
     || !(alpha > 0.0)
     || strength < 0.0
    )
    {
        return invalidParticleWallContactResult();
    }
    if (!finiteParticleWallContactValue(strength))
    {
        strength = DBL_MAX;
    }

    const double fraction = boundedContactFraction(strength);
    const double fullNewTemperature =
        particleTemperatureK
      - (particleTemperatureK - wallTemperatureK)*fraction;
    const double fullWallEnergy =
        parcelMassKg
       *(
            aluminaSpecificEnthalpyJkg(particleTemperatureK)
          - aluminaSpecificEnthalpyJkg(fullNewTemperature)
        );
    const double wallEnergy =
        depositionHeatTransferEfficiency*contactAreaScale*fullWallEnergy;
    const double newSpecificEnthalpy =
        aluminaSpecificEnthalpyJkg(particleTemperatureK)
      - wallEnergy/parcelMassKg;
    const double newTemperature =
        aluminaTemperatureFromSpecificEnthalpyK(newSpecificEnthalpy);
    if
    (
        !finiteParticleWallContactValue(newTemperature)
     || !finiteParticleWallContactValue(fullWallEnergy)
     || !finiteParticleWallContactValue(wallEnergy)
     || newTemperature <= 0.0
    )
    {
        return invalidParticleWallContactResult();
    }

    return
    {
        true,
        newTemperature,
        -wallEnergy,
        wallEnergy,
        fraction,
        strength,
        physicalMass,
        0.0,
        0.0,
        0.0,
        deltaTSeconds,
        0.0,
        depositionAreaM2
    };
}

}                        
}                  

#undef GPU_PARTICLE_WALL_CONTACT_HD

#endif
