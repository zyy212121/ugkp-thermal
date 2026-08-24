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

constexpr double aluminaPropertyTemperatureMinK = 2327.0;
constexpr double aluminaPropertyTemperatureMaxK = 4000.0;
constexpr double aluminaMolarMassKgMol = 0.101961276;
constexpr double aluminaCpAtMeltingMolarJMolK = 153.5;
constexpr double aluminaCpMolarSlopeJMolK2 = 3.1e-3;
constexpr double aluminaThermalConductivityWmK = 2.70;

struct AluminaLiquidProperties
{
    bool valid;
    double temperatureK;
    double densityKgM3;
    double viscosityPaS;
    double surfaceTensionNM;
    double specificHeatJkgK;
    double thermalConductivityWmK;
};

GPU_ALUMINA_HD inline bool finiteAluminaValue(const double value) noexcept
{
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

GPU_ALUMINA_HD inline double clampAlumina
(
    const double value,
    const double lower,
    const double upper
) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

GPU_ALUMINA_HD inline double aluminaSpecificHeatJkgK
(
    const double temperatureK
) noexcept
{
    if (!finiteAluminaValue(temperatureK) || !(temperatureK > 0.0))
    {
        return 0.0;
    }
    const double boundedTemperatureK = clampAlumina
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

GPU_ALUMINA_HD inline double aluminaSpecificEnthalpyJkg
(
    const double temperatureK
) noexcept
{
    if (!finiteAluminaValue(temperatureK) || temperatureK < 0.0)
    {
        return -1.0;
    }

    const double cpMin = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMinK
    );
    const double cpMax = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMaxK
    );
    const double cpSlope = aluminaCpMolarSlopeJMolK2/aluminaMolarMassKgMol;
    if (temperatureK <= aluminaPropertyTemperatureMinK)
    {
        return cpMin*temperatureK;
    }

    const double spanK =
        aluminaPropertyTemperatureMaxK - aluminaPropertyTemperatureMinK;
    const double hAtMin = cpMin*aluminaPropertyTemperatureMinK;
    if (temperatureK <= aluminaPropertyTemperatureMaxK)
    {
        const double deltaK =
            temperatureK - aluminaPropertyTemperatureMinK;
        return hAtMin + cpMin*deltaK + 0.5*cpSlope*deltaK*deltaK;
    }

    const double hAtMax =
        hAtMin + cpMin*spanK + 0.5*cpSlope*spanK*spanK;
    return
        hAtMax
      + cpMax*(temperatureK - aluminaPropertyTemperatureMaxK);
}

GPU_ALUMINA_HD inline double aluminaTemperatureFromSpecificEnthalpyK
(
    const double specificEnthalpyJkg
) noexcept
{
    if
    (
        !finiteAluminaValue(specificEnthalpyJkg)
     || specificEnthalpyJkg < 0.0
    )
    {
        return -1.0;
    }

    const double cpMin = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMinK
    );
    const double cpMax = aluminaSpecificHeatJkgK
    (
        aluminaPropertyTemperatureMaxK
    );
    const double cpSlope = aluminaCpMolarSlopeJMolK2/aluminaMolarMassKgMol;
    const double hAtMin = cpMin*aluminaPropertyTemperatureMinK;
    if (specificEnthalpyJkg <= hAtMin)
    {
        return specificEnthalpyJkg/cpMin;
    }

    const double spanK =
        aluminaPropertyTemperatureMaxK - aluminaPropertyTemperatureMinK;
    const double hAtMax =
        hAtMin + cpMin*spanK + 0.5*cpSlope*spanK*spanK;
    if (specificEnthalpyJkg <= hAtMax)
    {
        const double deltaH = specificEnthalpyJkg - hAtMin;
        const double root = ::sqrt(cpMin*cpMin + 2.0*cpSlope*deltaH);
        const double deltaK = 2.0*deltaH/(cpMin + root);
        return aluminaPropertyTemperatureMinK + deltaK;
    }

    return
        aluminaPropertyTemperatureMaxK
      + (specificEnthalpyJkg - hAtMax)/cpMax;
}

GPU_ALUMINA_HD inline AluminaLiquidProperties liquidAluminaProperties
(
    const double particleTemperatureK
) noexcept
{
    if (!finiteAluminaValue(particleTemperatureK) || !(particleTemperatureK > 0.0))
    {
        return {false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double temperatureK = clampAlumina
    (
        particleTemperatureK,
        aluminaPropertyTemperatureMinK,
        aluminaPropertyTemperatureMaxK
    );
    const double densityKgM3 =
        1000.0*(2.917 - 1.228e-4*(temperatureK - 2327.0));
    const double surfaceTensionNM =
        0.632 - 2.310e-5*(temperatureK - 2327.0);
    const double viscosityPaS =
        5.77e-4*::exp(9743.0/temperatureK);
    const double specificHeatJkgK = aluminaSpecificHeatJkgK(temperatureK);
    const double thermalConductivityWmK = aluminaThermalConductivityWmK;

    const bool valid =
        finiteAluminaValue(densityKgM3)
     && finiteAluminaValue(surfaceTensionNM)
     && finiteAluminaValue(viscosityPaS)
     && finiteAluminaValue(specificHeatJkgK)
     && densityKgM3 > 0.0
     && surfaceTensionNM > 0.0
     && viscosityPaS > 0.0
     && specificHeatJkgK > 0.0
     && thermalConductivityWmK > 0.0;
    return
    {
        valid,
        temperatureK,
        valid ? densityKgM3 : 0.0,
        valid ? viscosityPaS : 0.0,
        valid ? surfaceTensionNM : 0.0,
        valid ? specificHeatJkgK : 0.0,
        valid ? thermalConductivityWmK : 0.0
    };
}

}                        
}                  

#undef GPU_ALUMINA_HD

#endif
