#ifndef GPU_THERMAL_GPU_WALL_INTERFACE_HEAT_H
#define GPU_THERMAL_GPU_WALL_INTERFACE_HEAT_H

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define GPU_WALL_INTERFACE_HD __host__ __device__
#else
#define GPU_WALL_INTERFACE_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr double wallInterfacePi =
    3.141592653589793238462643383279502884;

GPU_WALL_INTERFACE_HD inline bool finiteWallInterfaceValue
(
    const double value
) noexcept
{
    return value == value && value >= -DBL_MAX && value <= DBL_MAX;
}

GPU_WALL_INTERFACE_HD inline double wallInterfaceLog1pMinusX
(
    const double x
) noexcept
{
    if (::fabs(x) >= 1.0e-4)
    {
        return ::log1p(x) - x;
    }
    const double x2 = x*x;
    const double x3 = x2*x;
    const double x4 = x3*x;
    const double x5 = x4*x;
    const double x6 = x5*x;
    return -0.5*x2 + x3/3.0 - 0.25*x4 + 0.2*x5 - x6/6.0;
}

GPU_WALL_INTERFACE_HD inline double wallInterfaceResistanceTimeIntegral
(
    const double age0S,
    const double age1S,
    const double baseResistanceM2KW,
    const double wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if
    (
        !finiteWallInterfaceValue(age0S)
     || !finiteWallInterfaceValue(age1S)
     || !finiteWallInterfaceValue(baseResistanceM2KW)
     || !finiteWallInterfaceValue(wallEffusivityWsHalfM2K)
     || age0S < 0.0
     || age1S < age0S
     || baseResistanceM2KW < 0.0
     || !(wallEffusivityWsHalfM2K > 0.0)
    )
    {
        return -1.0;
    }
    const double deltaAge = age1S - age0S;
    if (!(deltaAge > 0.0))
    {
        return 0.0;
    }
    if (!wallTransientResistance)
    {
        return baseResistanceM2KW > 0.0
          ? deltaAge/baseResistanceM2KW
          : -1.0;
    }
    const double c = ::sqrt(wallInterfacePi)/wallEffusivityWsHalfM2K;
    const double root0 = ::sqrt(age0S);
    const double root1 = ::sqrt(age1S);
    const double rootDifference = root1 - root0;
    if (!(c > 0.0) || !finiteWallInterfaceValue(c))
    {
        return -1.0;
    }
    if (!(baseResistanceM2KW > 0.0))
    {
        const double integral = 2.0*rootDifference/c;
        return finiteWallInterfaceValue(integral) && integral >= 0.0
          ? integral
          : -1.0;
    }
    const double x0 = c*root0;
    const double dx = c*rootDifference;
    const double denominator0 = baseResistanceM2KW + x0;
    const double ratioIncrement = dx/denominator0;
    const double bracket =
        x0*ratioIncrement
      - baseResistanceM2KW*wallInterfaceLog1pMinusX(ratioIncrement);
    const double integral = 2.0*bracket/(c*c);
    return finiteWallInterfaceValue(integral) && integral >= 0.0
      ? integral
      : -1.0;
}

GPU_WALL_INTERFACE_HD inline double wallInterfaceConductanceTimeIntegral
(
    const double areaM2,
    const double efficiency,
    const double age0S,
    const double age1S,
    const double interfaceResistanceM2KW,
    const double particleAdditionalResistanceM2KW,
    const double wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if
    (
        !finiteWallInterfaceValue(areaM2)
     || !finiteWallInterfaceValue(efficiency)
     || !finiteWallInterfaceValue(interfaceResistanceM2KW)
     || !finiteWallInterfaceValue(particleAdditionalResistanceM2KW)
     || !(areaM2 >= 0.0)
     || !(efficiency > 0.0)
     || efficiency > 1.0
     || interfaceResistanceM2KW < 0.0
     || particleAdditionalResistanceM2KW < 0.0
    )
    {
        return -1.0;
    }
    if (!(areaM2 > 0.0))
    {
        return 0.0;
    }
    const double resistanceIntegral = wallInterfaceResistanceTimeIntegral
    (
        age0S,
        age1S,
        interfaceResistanceM2KW + particleAdditionalResistanceM2KW,
        wallEffusivityWsHalfM2K,
        wallTransientResistance
    );
    const double conductanceIntegral =
        efficiency*areaM2*resistanceIntegral;
    return
        resistanceIntegral >= 0.0
     && finiteWallInterfaceValue(conductanceIntegral)
     && conductanceIntegral >= 0.0
      ? conductanceIntegral
      : -1.0;
}

}
}

#undef GPU_WALL_INTERFACE_HD

#endif
