#include "GpuPrecisionTypes.H"
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

constexpr GpuReal wallInterfacePi =
    GPU_R(3.141592653589793238462643383279502884);

GPU_WALL_INTERFACE_HD inline bool finiteWallInterfaceValue
(
    const GpuReal value
) noexcept
{
    return value == value && value >= -GPU_REAL_MAX && value <= GPU_REAL_MAX;
}

GPU_WALL_INTERFACE_HD inline GpuReal wallInterfaceLog1pMinusX
(
    const GpuReal x
) noexcept
{
    if (::fabs(x) >= GPU_R(1.0e-4))
    {
        return ::log1p(x) - x;
    }
    const GpuReal x2 = x*x;
    const GpuReal x3 = x2*x;
    const GpuReal x4 = x3*x;
    const GpuReal x5 = x4*x;
    const GpuReal x6 = x5*x;
    return -GPU_R(0.5)*x2 + x3/GPU_R(3.0) - GPU_R(0.25)*x4 + GPU_R(0.2)*x5 - x6/GPU_R(6.0);
}

GPU_WALL_INTERFACE_HD inline GpuReal wallInterfaceResistanceTimeIntegral
(
    const GpuTime age0S,
    const GpuTime age1S,
    const GpuReal baseResistanceM2KW,
    const GpuReal wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if
    (
        !finiteWallInterfaceValue(age0S)
     || !finiteWallInterfaceValue(age1S)
     || !finiteWallInterfaceValue(baseResistanceM2KW)
     || !finiteWallInterfaceValue(wallEffusivityWsHalfM2K)
     || age0S < GPU_R(0.0)
     || age1S < age0S
     || baseResistanceM2KW < GPU_R(0.0)
     || !(wallEffusivityWsHalfM2K > GPU_R(0.0))
    )
    {
        return -GPU_R(1.0);
    }
    const GpuTime deltaAge = age1S - age0S;
    if (!(deltaAge > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    if (!wallTransientResistance)
    {
        return baseResistanceM2KW > GPU_R(0.0)
          ? deltaAge/baseResistanceM2KW
          : -GPU_R(1.0);
    }
    const GpuReal c = ::sqrt(wallInterfacePi)/wallEffusivityWsHalfM2K;
    const GpuTime root0 = ::sqrt(age0S);
    const GpuTime root1 = ::sqrt(age1S);
    #if UGKWP_GPU_REAL_BITS == 32
    const GpuTime rootDifference = deltaAge/(root1 + root0);
#else
    const GpuTime rootDifference = root1 - root0;
#endif
    if (!(c > GPU_R(0.0)) || !finiteWallInterfaceValue(c))
    {
        return -GPU_R(1.0);
    }
    if (!(baseResistanceM2KW > GPU_R(0.0)))
    {
        const GpuReal integral = GPU_R(2.0)*rootDifference/c;
        return finiteWallInterfaceValue(integral) && integral >= GPU_R(0.0)
          ? integral
          : -GPU_R(1.0);
    }
    const GpuReal x0 = c*root0;
    const GpuReal dx = c*rootDifference;
    const GpuReal denominator0 = baseResistanceM2KW + x0;
    const GpuReal ratioIncrement = dx/denominator0;
    const GpuReal bracket =
        x0*ratioIncrement
      - baseResistanceM2KW*wallInterfaceLog1pMinusX(ratioIncrement);
    const GpuReal integral = GPU_R(2.0)*bracket/(c*c);
    return finiteWallInterfaceValue(integral) && integral >= GPU_R(0.0)
      ? integral
      : -GPU_R(1.0);
}

GPU_WALL_INTERFACE_HD inline GpuReal wallInterfaceConductanceTimeIntegral
(
    const GpuReal areaM2,
    const GpuReal efficiency,
    const GpuTime age0S,
    const GpuTime age1S,
    const GpuReal interfaceResistanceM2KW,
    const GpuReal particleAdditionalResistanceM2KW,
    const GpuReal wallEffusivityWsHalfM2K,
    const bool wallTransientResistance
) noexcept
{
    if
    (
        !finiteWallInterfaceValue(areaM2)
     || !finiteWallInterfaceValue(efficiency)
     || !finiteWallInterfaceValue(interfaceResistanceM2KW)
     || !finiteWallInterfaceValue(particleAdditionalResistanceM2KW)
     || !(areaM2 >= GPU_R(0.0))
     || !(efficiency > GPU_R(0.0))
     || efficiency > GPU_R(1.0)
     || interfaceResistanceM2KW < GPU_R(0.0)
     || particleAdditionalResistanceM2KW < GPU_R(0.0)
    )
    {
        return -GPU_R(1.0);
    }
    if (!(areaM2 > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }
    const GpuReal resistanceIntegral = wallInterfaceResistanceTimeIntegral
    (
        age0S,
        age1S,
        interfaceResistanceM2KW + particleAdditionalResistanceM2KW,
        wallEffusivityWsHalfM2K,
        wallTransientResistance
    );
    const GpuReal conductanceIntegral =
        efficiency*areaM2*resistanceIntegral;
    return
        resistanceIntegral >= GPU_R(0.0)
     && finiteWallInterfaceValue(conductanceIntegral)
     && conductanceIntegral >= GPU_R(0.0)
      ? conductanceIntegral
      : -GPU_R(1.0);
}

}
}

#undef GPU_WALL_INTERFACE_HD

#endif
