#ifndef GPU_THERMAL_GPU_CAPILLARY_DETACHMENT_H
#define GPU_THERMAL_GPU_CAPILLARY_DETACHMENT_H

#include "GpuAluminaProperties.H"

#include <cmath>

#if defined(__CUDACC__)
#define GPU_CAPILLARY_HD __host__ __device__
#else
#define GPU_CAPILLARY_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr double capillaryPi =
    3.141592653589793238462643383279502884;

struct CapillaryDetachmentState
{
    bool valid;
    double adhesionSpecificEnergyJkg;
    double contactAngleCosine;
    double equilibriumContactAreaM2;
    AluminaLiquidProperties material;
};

struct CapillaryContactDamageResult
{
    bool valid;
    bool detached;
    double remainingContactAreaM2;
    double residualSpecificEnergyJkg;
};

GPU_CAPILLARY_HD inline bool finiteCapillaryValue
(
    const double value
) noexcept
{
    return finiteAluminaValue(value);
}

GPU_CAPILLARY_HD inline double clampCapillary
(
    const double value,
    const double lower,
    const double upper
) noexcept
{
    return clampAlumina(value, lower, upper);
}

GPU_CAPILLARY_HD inline CapillaryDetachmentState
evaluateCapillaryDetachmentState
(
    const double particleTemperatureK,
    const double particleDiameterM,
    const double adhesionEnergyScale,
    const double contactAngleCosine
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if
    (
        !material.valid
     || !finiteCapillaryValue(particleDiameterM)
     || !finiteCapillaryValue(adhesionEnergyScale)
     || !finiteCapillaryValue(contactAngleCosine)
     || !(particleDiameterM > 0.0)
     || !(adhesionEnergyScale > 0.0)
    )
    {
        return {false, 0.0, 0.0, 0.0, material};
    }

    const double cosine = clampCapillary
    (
        contactAngleCosine,
        -1.0,
        1.0
    );
    const double diameter3 =
        particleDiameterM*particleDiameterM*particleDiameterM;
    const double volumeM3 = capillaryPi*diameter3/6.0;
    const double capDenominator = 2.0 - 3.0*cosine + cosine*cosine*cosine;
    if
    (
        !finiteCapillaryValue(capDenominator)
     || !(volumeM3 > 0.0)
     || !(capDenominator > 0.0)
    )
    {
        return {false, 0.0, 0.0, 0.0, material};
    }

    const double capRadiusM =
        ::cbrt(3.0*volumeM3/(capillaryPi*capDenominator));
    const double contactAreaM2 =
        capillaryPi*capRadiusM*capRadiusM*(1.0 - cosine*cosine);
    const double workOfAdhesionJM2 =
        material.surfaceTensionNM*(1.0 + cosine);
    const double physicalMassKg = material.densityKgM3*volumeM3;
    const double adhesionSpecificEnergyJkg =
        adhesionEnergyScale*workOfAdhesionJM2*contactAreaM2/physicalMassKg;

    const bool valid =
        finiteCapillaryValue(contactAreaM2)
     && finiteCapillaryValue(adhesionSpecificEnergyJkg)
     && contactAreaM2 > 0.0
     && adhesionSpecificEnergyJkg > 0.0;
    return
    {
        valid,
        valid ? adhesionSpecificEnergyJkg : 0.0,
        cosine,
        valid ? contactAreaM2 : 0.0,
        material
    };
}

GPU_CAPILLARY_HD inline CapillaryContactDamageResult
applyCapillaryContactDamage
(
    const CapillaryDetachmentState& capillary,
    const double currentContactAreaM2,
    const double contactAreaScale,
    const double sampledSpecificEnergyJkg
) noexcept
{
    if
    (
        !capillary.valid
     || !finiteCapillaryValue(currentContactAreaM2)
     || !finiteCapillaryValue(contactAreaScale)
     || !finiteCapillaryValue(sampledSpecificEnergyJkg)
     || !(capillary.equilibriumContactAreaM2 > 0.0)
     || !(capillary.adhesionSpecificEnergyJkg > 0.0)
     || !(currentContactAreaM2 > 0.0)
     || !(contactAreaScale > 0.0)
     || contactAreaScale > 1.0
     || sampledSpecificEnergyJkg < 0.0
    )
    {
        return {false, false, 0.0, 0.0};
    }

    const double adhesionSpecificEnergyPerArea =
        capillary.adhesionSpecificEnergyJkg
       /capillary.equilibriumContactAreaM2;
    const double remainingAdhesionSpecificEnergy =
        contactAreaScale*adhesionSpecificEnergyPerArea*currentContactAreaM2;
    if
    (
        !finiteCapillaryValue(adhesionSpecificEnergyPerArea)
     || !finiteCapillaryValue(remainingAdhesionSpecificEnergy)
     || !(adhesionSpecificEnergyPerArea > 0.0)
     || !(remainingAdhesionSpecificEnergy > 0.0)
    )
    {
        return {false, false, 0.0, 0.0};
    }

    if (sampledSpecificEnergyJkg >= remainingAdhesionSpecificEnergy)
    {
        const double residual =
            sampledSpecificEnergyJkg - remainingAdhesionSpecificEnergy;
        return
        {
            finiteCapillaryValue(residual),
            true,
            0.0,
            finiteCapillaryValue(residual) ? residual : 0.0
        };
    }

    const double consumedAreaM2 =
        sampledSpecificEnergyJkg
       /(contactAreaScale*adhesionSpecificEnergyPerArea);
    const double remainingAreaM2 =
        currentContactAreaM2 - consumedAreaM2;
    const bool valid =
        finiteCapillaryValue(consumedAreaM2)
     && finiteCapillaryValue(remainingAreaM2)
     && consumedAreaM2 >= 0.0
     && remainingAreaM2 > 0.0;
    return
    {
        valid,
        false,
        valid ? remainingAreaM2 : 0.0,
        0.0
    };
}

}                        
}                  

#undef GPU_CAPILLARY_HD

#endif
