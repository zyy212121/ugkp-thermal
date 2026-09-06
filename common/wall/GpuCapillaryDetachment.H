#include "GpuPrecisionTypes.H"
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

constexpr GpuReal capillaryPi =
    GPU_R(3.141592653589793238462643383279502884);

struct CapillaryDetachmentState
{
    bool valid;
    GpuReal adhesionSpecificEnergyJkg;
    GpuReal contactAngleCosine;
    GpuReal equilibriumContactAreaM2;
    AluminaLiquidProperties material;
};

struct CapillaryContactDamageResult
{
    bool valid;
    bool detached;
    GpuReal remainingContactAreaM2;
    GpuReal residualSpecificEnergyJkg;
};

GPU_CAPILLARY_HD inline bool finiteCapillaryValue
(
    const GpuReal value
) noexcept
{
    return finiteAluminaValue(value);
}

GPU_CAPILLARY_HD inline GpuReal clampCapillary
(
    const GpuReal value,
    const GpuReal lower,
    const GpuReal upper
) noexcept
{
    return clampAlumina(value, lower, upper);
}

GPU_CAPILLARY_HD inline CapillaryDetachmentState
evaluateCapillaryDetachmentState
(
    const GpuReal particleTemperatureK,
    const GpuReal particleDiameterM,
    const GpuReal adhesionEnergyScale,
    const GpuReal contactAngleCosine
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
     || !(particleDiameterM > GPU_R(0.0))
     || !(adhesionEnergyScale > GPU_R(0.0))
    )
    {
        return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), material};
    }

    const GpuReal cosine = clampCapillary
    (
        contactAngleCosine,
        -GPU_R(1.0),
        GPU_R(1.0)
    );
    const GpuReal diameter3 =
        particleDiameterM*particleDiameterM*particleDiameterM;
    const GpuReal volumeM3 = capillaryPi*diameter3/GPU_R(6.0);
    const GpuReal capDenominator = GPU_R(2.0) - GPU_R(3.0)*cosine + cosine*cosine*cosine;
    if
    (
        !finiteCapillaryValue(capDenominator)
     || !(volumeM3 > GPU_R(0.0))
     || !(capDenominator > GPU_R(0.0))
    )
    {
        return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), material};
    }

    const GpuReal capRadiusM =
        ::cbrt(GPU_R(3.0)*volumeM3/(capillaryPi*capDenominator));
    const GpuReal contactAreaM2 =
        capillaryPi*capRadiusM*capRadiusM*(GPU_R(1.0) - cosine*cosine);
    const GpuReal workOfAdhesionJM2 =
        material.surfaceTensionNM*(GPU_R(1.0) + cosine);
    const GpuReal physicalMassKg = material.densityKgM3*volumeM3;
    const GpuReal adhesionSpecificEnergyJkg =
        adhesionEnergyScale*workOfAdhesionJM2*contactAreaM2/physicalMassKg;

    const bool valid =
        finiteCapillaryValue(contactAreaM2)
     && finiteCapillaryValue(adhesionSpecificEnergyJkg)
     && contactAreaM2 > GPU_R(0.0)
     && adhesionSpecificEnergyJkg > GPU_R(0.0);
    return
    {
        valid,
        valid ? adhesionSpecificEnergyJkg : GPU_R(0.0),
        cosine,
        valid ? contactAreaM2 : GPU_R(0.0),
        material
    };
}

GPU_CAPILLARY_HD inline CapillaryContactDamageResult
applyCapillaryContactDamage
(
    const CapillaryDetachmentState& capillary,
    const GpuReal currentContactAreaM2,
    const GpuReal contactAreaScale,
    const GpuReal sampledSpecificEnergyJkg
) noexcept
{
    if
    (
        !capillary.valid
     || !finiteCapillaryValue(currentContactAreaM2)
     || !finiteCapillaryValue(contactAreaScale)
     || !finiteCapillaryValue(sampledSpecificEnergyJkg)
     || !(capillary.equilibriumContactAreaM2 > GPU_R(0.0))
     || !(capillary.adhesionSpecificEnergyJkg > GPU_R(0.0))
     || !(currentContactAreaM2 > GPU_R(0.0))
     || !(contactAreaScale > GPU_R(0.0))
     || contactAreaScale > GPU_R(1.0)
     || sampledSpecificEnergyJkg < GPU_R(0.0)
    )
    {
        return {false, false, GPU_R(0.0), GPU_R(0.0)};
    }

    const GpuReal adhesionSpecificEnergyPerArea =
        capillary.adhesionSpecificEnergyJkg
       /capillary.equilibriumContactAreaM2;
    const GpuReal remainingAdhesionSpecificEnergy =
        contactAreaScale*adhesionSpecificEnergyPerArea*currentContactAreaM2;
    if
    (
        !finiteCapillaryValue(adhesionSpecificEnergyPerArea)
     || !finiteCapillaryValue(remainingAdhesionSpecificEnergy)
     || !(adhesionSpecificEnergyPerArea > GPU_R(0.0))
     || !(remainingAdhesionSpecificEnergy > GPU_R(0.0))
    )
    {
        return {false, false, GPU_R(0.0), GPU_R(0.0)};
    }

    if (sampledSpecificEnergyJkg >= remainingAdhesionSpecificEnergy)
    {
        const GpuReal residual =
            sampledSpecificEnergyJkg - remainingAdhesionSpecificEnergy;
        return
        {
            finiteCapillaryValue(residual),
            true,
            GPU_R(0.0),
            finiteCapillaryValue(residual) ? residual : GPU_R(0.0)
        };
    }

    const GpuReal consumedAreaM2 =
        sampledSpecificEnergyJkg
       /(contactAreaScale*adhesionSpecificEnergyPerArea);
    const GpuReal remainingAreaM2 =
        currentContactAreaM2 - consumedAreaM2;
    const bool valid =
        finiteCapillaryValue(consumedAreaM2)
     && finiteCapillaryValue(remainingAreaM2)
     && consumedAreaM2 >= GPU_R(0.0)
     && remainingAreaM2 > GPU_R(0.0);
    return
    {
        valid,
        false,
        valid ? remainingAreaM2 : GPU_R(0.0),
        GPU_R(0.0)
    };
}

}                        
}                  

#undef GPU_CAPILLARY_HD

#endif
