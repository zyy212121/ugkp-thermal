#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_ALUMINA_PROPERTIES_H
#define GPU_THERMAL_GPU_ALUMINA_PROPERTIES_H

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define GPU_ALUMINA_HD __host__ __device__
#else
#define GPU_ALUMINA_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr GpuReal aluminaPropertyTemperatureMinK = GPU_R(2327.0);
constexpr GpuReal aluminaPropertyTemperatureMaxK = GPU_R(4000.0);
constexpr GpuReal aluminaMolarMassKgMol = GPU_R(0.101961276);
constexpr GpuReal aluminaCpAtMeltingMolarJMolK = GPU_R(153.5);
constexpr GpuReal aluminaCpMolarSlopeJMolK2 = GPU_R(3.1e-3);
constexpr GpuReal aluminaThermalConductivityWmK = GPU_R(2.70);

struct AluminaLiquidProperties
{
    bool valid;
    GpuReal temperatureK;
    GpuReal densityKgM3;
    GpuReal viscosityPaS;
    GpuReal surfaceTensionNM;
    GpuReal specificHeatJkgK;
    GpuReal thermalConductivityWmK;
};

GPU_ALUMINA_HD inline bool finiteAluminaValue(const GpuReal value) noexcept
{
    return value == value && value >= -GPU_REAL_MAX && value <= GPU_REAL_MAX;
}

GPU_ALUMINA_HD inline GpuReal clampAlumina
(
    const GpuReal value,
    const GpuReal lower,
    const GpuReal upper
) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

GPU_ALUMINA_HD inline GpuReal aluminaSpecificHeatJkgK
(
    const GpuReal temperatureK
) noexcept
{
    if (!finiteAluminaValue(temperatureK) || !(temperatureK > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    const GpuReal boundedTemperatureK = clampAlumina
    (
        temperatureK,
        aluminaPropertyTemperatureMinK,
        aluminaPropertyTemperatureMaxK
    );
    return
    (
        aluminaCpAtMeltingMolarJMolK
      + aluminaCpMolarSlopeJMolK2
       *(boundedTemperatureK - aluminaPropertyTemperatureMinK)
    )/aluminaMolarMassKgMol;
}

GPU_ALUMINA_HD inline GpuReal aluminaSpecificEnthalpyJkg
(
    const GpuReal temperatureK
) noexcept
{
    if (!finiteAluminaValue(temperatureK) || temperatureK < GPU_R(0.0))
    {
        return -GPU_R(1.0);
    }

    const GpuReal cpMin = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMinK
    );
    const GpuReal cpMax = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMaxK
    );
    const GpuReal cpSlope = aluminaCpMolarSlopeJMolK2/aluminaMolarMassKgMol;
    if (temperatureK <= aluminaPropertyTemperatureMinK)
    {
        return cpMin*temperatureK;
    }

    const GpuReal spanK =
        aluminaPropertyTemperatureMaxK - aluminaPropertyTemperatureMinK;
    const GpuReal hAtMin = cpMin*aluminaPropertyTemperatureMinK;
    if (temperatureK <= aluminaPropertyTemperatureMaxK)
    {
        const GpuReal deltaK =
            temperatureK - aluminaPropertyTemperatureMinK;
        return hAtMin + cpMin*deltaK + GPU_R(0.5)*cpSlope*deltaK*deltaK;
    }

    const GpuReal hAtMax =
        hAtMin + cpMin*spanK + GPU_R(0.5)*cpSlope*spanK*spanK;
    return
        hAtMax
      + cpMax*(temperatureK - aluminaPropertyTemperatureMaxK);
}

GPU_ALUMINA_HD inline GpuReal aluminaTemperatureFromSpecificEnthalpyK
(
    const GpuReal specificEnthalpyJkg
) noexcept
{
    if
    (
        !finiteAluminaValue(specificEnthalpyJkg)
     || specificEnthalpyJkg < GPU_R(0.0)
    )
    {
        return -GPU_R(1.0);
    }

    const GpuReal cpMin = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMinK
    );
    const GpuReal cpMax = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMaxK
    );
    const GpuReal cpSlope = aluminaCpMolarSlopeJMolK2/aluminaMolarMassKgMol;
    const GpuReal hAtMin = cpMin*aluminaPropertyTemperatureMinK;
    if (specificEnthalpyJkg <= hAtMin)
    {
        return specificEnthalpyJkg/cpMin;
    }

    const GpuReal spanK =
        aluminaPropertyTemperatureMaxK - aluminaPropertyTemperatureMinK;
    const GpuReal hAtMax =
        hAtMin + cpMin*spanK + GPU_R(0.5)*cpSlope*spanK*spanK;
    if (specificEnthalpyJkg <= hAtMax)
    {
        const GpuReal deltaH = specificEnthalpyJkg - hAtMin;
        const GpuReal root = ::sqrt(cpMin*cpMin + GPU_R(2.0)*cpSlope*deltaH);
        const GpuReal deltaK = GPU_R(2.0)*deltaH/(cpMin + root);
        return aluminaPropertyTemperatureMinK + deltaK;
    }

    return
        aluminaPropertyTemperatureMaxK
      + (specificEnthalpyJkg - hAtMax)/cpMax;
}

GPU_ALUMINA_HD inline AluminaLiquidProperties liquidAluminaProperties
(
    const GpuReal particleTemperatureK
) noexcept
{
    if (!finiteAluminaValue(particleTemperatureK) || !(particleTemperatureK > GPU_R(0.0)))
    {
        return {false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    }

    const GpuReal temperatureK = clampAlumina
    (
        particleTemperatureK,
        aluminaPropertyTemperatureMinK,
        aluminaPropertyTemperatureMaxK
    );
    const GpuReal densityKgM3 =
        GPU_R(1000.0)*(GPU_R(2.917) - GPU_R(1.228e-4)*(temperatureK - GPU_R(2327.0)));
    const GpuReal surfaceTensionNM =
        GPU_R(0.632) - GPU_R(2.310e-5)*(temperatureK - GPU_R(2327.0));
    const GpuReal viscosityPaS =
        GPU_R(5.77e-4)*::exp(GPU_R(9743.0)/temperatureK);
    const GpuReal specificHeatJkgK = aluminaSpecificHeatJkgK(temperatureK);
    const GpuReal thermalConductivityWmK = aluminaThermalConductivityWmK;

    const bool valid =
        finiteAluminaValue(densityKgM3)
     && finiteAluminaValue(surfaceTensionNM)
     && finiteAluminaValue(viscosityPaS)
     && finiteAluminaValue(specificHeatJkgK)
     && densityKgM3 > GPU_R(0.0)
     && surfaceTensionNM > GPU_R(0.0)
     && viscosityPaS > GPU_R(0.0)
     && specificHeatJkgK > GPU_R(0.0)
     && thermalConductivityWmK > GPU_R(0.0);
    return
    {
        valid,
        temperatureK,
        valid ? densityKgM3 : GPU_R(0.0),
        valid ? viscosityPaS : GPU_R(0.0),
        valid ? surfaceTensionNM : GPU_R(0.0),
        valid ? specificHeatJkgK : GPU_R(0.0),
        valid ? thermalConductivityWmK : GPU_R(0.0)
    };
}

}                        
}                  

#undef GPU_ALUMINA_HD

#endif
