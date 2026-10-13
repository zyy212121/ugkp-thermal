#include "GpuPrecisionTypes.H"
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
    GpuReal weber;
    GpuReal reynolds;
    GpuReal sommerfeld;
};

GPU_SOMMERFELD_HD inline bool finiteSommerfeldValue
(
    const GpuReal value
) noexcept
{
    return value == value && value >= -GPU_REAL_MAX && value <= GPU_REAL_MAX;
}

GPU_SOMMERFELD_HD inline SommerfeldImpact evaluateSommerfeldImpact
(
    const GpuReal particleTemperatureK,
    const GpuReal normalSpeed,
    const GpuReal particleDiameter,
    const GpuReal threshold
) noexcept
{
    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);

    const GpuReal weber =
        material.densityKgM3*normalSpeed*normalSpeed*particleDiameter
       /material.surfaceTensionNM;
    const GpuReal reynolds =
        material.densityKgM3*normalSpeed*particleDiameter
       /material.viscosityPaS;
    const GpuReal sommerfeld = ::sqrt(weber)*::pow(reynolds, GPU_R(0.25));

    return
    {
        finiteSommerfeldValue(weber) && finiteSommerfeldValue(reynolds)
         && finiteSommerfeldValue(sommerfeld),
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
