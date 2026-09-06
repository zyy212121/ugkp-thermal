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
    if
    (
        !finiteSommerfeldValue(particleTemperatureK)
     || !finiteSommerfeldValue(normalSpeed)
     || !finiteSommerfeldValue(particleDiameter)
     || !finiteSommerfeldValue(threshold)
     || !(particleTemperatureK > GPU_R(0.0))
     || normalSpeed < GPU_R(0.0)
     || !(particleDiameter > GPU_R(0.0))
     || !(threshold > GPU_R(0.0))
    )
    {
        return {false, false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    }

    const AluminaLiquidProperties material =
        liquidAluminaProperties(particleTemperatureK);
    if (!material.valid)
    {
        return {false, false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    }

    const GpuReal weber =
        material.densityKgM3*normalSpeed*normalSpeed*particleDiameter
       /material.surfaceTensionNM;
    const GpuReal reynolds =
        material.densityKgM3*normalSpeed*particleDiameter
       /material.viscosityPaS;
    const GpuReal sommerfeld = ::sqrt(weber)*::pow(reynolds, GPU_R(0.25));
    if
    (
        !finiteSommerfeldValue(weber)
     || !finiteSommerfeldValue(reynolds)
     || !finiteSommerfeldValue(sommerfeld)
     || weber < GPU_R(0.0)
     || reynolds < GPU_R(0.0)
     || sommerfeld < GPU_R(0.0)
    )
    {
        return {false, false, GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
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
