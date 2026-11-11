#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_ALUMINA_PROPERTIES_H
#define GPU_THERMAL_GPU_ALUMINA_PROPERTIES_H

#include <cmath>
#include <limits>
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

template<class Real>
GPU_ALUMINA_HD inline bool finiteAluminaValue(const Real value) noexcept
{
    return value == value && value >= -std::numeric_limits<Real>::max()
        && value <= std::numeric_limits<Real>::max();
}

template<class Real>
GPU_ALUMINA_HD inline Real clampAlumina
(
    const Real value,
    const Real lower,
    const Real upper
) noexcept
{
    return value < lower ? lower : (value > upper ? upper : value);
}

template<class Real>
GPU_ALUMINA_HD inline Real aluminaSpecificHeatJkgK
(
    const Real temperatureK
) noexcept
{
    const Real boundedTemperatureK = clampAlumina
    (
        temperatureK,
        Real(2327.0),
        Real(4000.0)
    );
    return
    (
        Real(153.5)
      + Real(3.1e-3)
       *(boundedTemperatureK - Real(2327.0))
    )/Real(0.101961276);
}

template<class Real>
GPU_ALUMINA_HD inline Real aluminaSpecificEnthalpyJkg
(
    const Real temperatureK
) noexcept
{
    const Real cpMin = aluminaSpecificHeatJkgK
    (
        Real(2327.0)
    );
    const Real cpMax = aluminaSpecificHeatJkgK
    (
        Real(4000.0)
    );
    const Real cpSlope = Real(3.1e-3)/Real(0.101961276);
    if (temperatureK <= Real(2327.0))
    {
        return cpMin*temperatureK;
    }

    const Real spanK =
        Real(4000.0) - Real(2327.0);
    const Real hAtMin = cpMin*Real(2327.0);
    if (temperatureK <= Real(4000.0))
    {
        const Real deltaK =
            temperatureK - Real(2327.0);
        return hAtMin + cpMin*deltaK + Real(0.5)*cpSlope*deltaK*deltaK;
    }

    const Real hAtMax =
        hAtMin + cpMin*spanK + Real(0.5)*cpSlope*spanK*spanK;
    return
        hAtMax
      + cpMax*(temperatureK - Real(4000.0));
}

template<class Real>
GPU_ALUMINA_HD inline Real aluminaTemperatureFromSpecificEnthalpyK
(
    const Real specificEnthalpyJkg
) noexcept
{
    const Real cpMin = aluminaSpecificHeatJkgK
    (
        Real(2327.0)
    );
    const Real cpMax = aluminaSpecificHeatJkgK
    (
        Real(4000.0)
    );
    const Real cpSlope = Real(3.1e-3)/Real(0.101961276);
    const Real hAtMin = cpMin*Real(2327.0);
    if (specificEnthalpyJkg <= hAtMin)
    {
        return specificEnthalpyJkg/cpMin;
    }

    const Real spanK =
        Real(4000.0) - Real(2327.0);
    const Real hAtMax =
        hAtMin + cpMin*spanK + Real(0.5)*cpSlope*spanK*spanK;
    if (specificEnthalpyJkg <= hAtMax)
    {
        const Real deltaH = specificEnthalpyJkg - hAtMin;
        const Real root = std::sqrt(cpMin*cpMin + Real(2.0)*cpSlope*deltaH);
        const Real deltaK = Real(2.0)*deltaH/(cpMin + root);
        return Real(2327.0) + deltaK;
    }

    return
        Real(4000.0)
      + (specificEnthalpyJkg - hAtMax)/cpMax;
}

GPU_ALUMINA_HD inline AluminaLiquidProperties liquidAluminaProperties
(
    const GpuReal particleTemperatureK
) noexcept
{
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
        densityKgM3,
        viscosityPaS,
        surfaceTensionNM,
        specificHeatJkgK,
        thermalConductivityWmK
    };
}

}                        
}                  

#undef GPU_ALUMINA_HD

#endif
