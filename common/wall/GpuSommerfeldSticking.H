#ifndef GPU_THERMAL_GPU_SOMMERFELD_STICKING_H
#define GPU_THERMAL_GPU_SOMMERFELD_STICKING_H

#include "GpuAluminaProperties.H"

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define GPU_SOMMERFELD_HD __host__ __device__
#else
#define GPU_SOMMERFELD_HD
#endif

namespace Foam
{
namespace gpuThermal
{

struct SommerfeldImpact
{
    bool valid;
    bool deposit;
    double weber;
    double reynolds;
    double sommerfeld;
};

GPU_SOMMERFELD_HD inline bool finiteSommerfeldValue
(
    const double value
) noexcept
{
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

GPU_SOMMERFELD_HD inline SommerfeldImpact evaluateSommerfeldImpact
(
    const double particleTemperatureK,
    const double normalSpeed,
    const double particleDiameter,
    const double threshold
) noexcept
{
    if
    (
        !finiteSommerfeldValue(particleTemperatureK)
     || !finiteSommerfeldValue(normalSpeed)
     || !finiteSommerfeldValue(particleDiameter)
     || !finiteSommerfeldValue(threshold)
     || !(particleTemperatureK > 0.0)
     || normalSpeed < 0.0
     || !(particleDiameter > 0.0)
     || !(threshold > 0.0)
    )
    {
        return {false, false, 0.0, 0.0, 0.0};
    }

    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if (!material.valid)
    {
        return {false, false, 0.0, 0.0, 0.0};
    }

    const double weber =
        material.densityKgM3*normalSpeed*normalSpeed*particleDiameter
       /material.surfaceTensionNM;
    const double reynolds =
        material.densityKgM3*normalSpeed*particleDiameter
       /material.viscosityPaS;
    const double sommerfeld = ::sqrt(weber)*::pow(reynolds, 0.25);
    if
    (
        !finiteSommerfeldValue(weber)
     || !finiteSommerfeldValue(reynolds)
     || !finiteSommerfeldValue(sommerfeld)
     || weber < 0.0
     || reynolds < 0.0
     || sommerfeld < 0.0
    )
    {
        return {false, false, 0.0, 0.0, 0.0};
    }
    return
    {
        true,
        sommerfeld >= threshold,
        weber,
        reynolds,
        sommerfeld
    };
}

}                        
}                  

#undef GPU_SOMMERFELD_HD

#endif
