#include "GpuPrecisionTypes.H"
#ifndef GPU_THERMAL_GPU_WALL_INTERFACE_HEAT_H
#define GPU_THERMAL_GPU_WALL_INTERFACE_HEAT_H

#include <cmath>
#include <limits>
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

template<class Real>
GPU_WALL_INTERFACE_HD inline bool finiteWallInterfaceValue(const Real value) noexcept
{
    return value == value && value >= -std::numeric_limits<Real>::max()
        && value <= std::numeric_limits<Real>::max();
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
    const GpuTime deltaAge = age1S - age0S;
    if (deltaAge == GPU_R(0.0))
    {
        return GPU_R(0.0);
    }
    if (!wallTransientResistance)
    {
        return deltaAge/baseResistanceM2KW;
    }
    const GpuReal c = ::sqrt(wallInterfacePi)/wallEffusivityWsHalfM2K;
    const GpuTime root0 = ::sqrt(age0S);
    const GpuTime root1 = ::sqrt(age1S);
    const GpuTime rootDifference = root1 - root0;

    if (baseResistanceM2KW == GPU_R(0.0))
    {
        const GpuReal integral = GPU_R(2.0)*rootDifference/c;
        return integral;
    }
    const GpuReal x0 = c*root0;
    const GpuReal dx = c*rootDifference;
    const GpuReal denominator0 = baseResistanceM2KW + x0;
    const GpuReal ratioIncrement = dx/denominator0;
    const GpuReal bracket =
        x0*ratioIncrement
      - baseResistanceM2KW*wallInterfaceLog1pMinusX(ratioIncrement);
    const GpuReal integral = GPU_R(2.0)*bracket/(c*c);
    return integral;
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
    if (areaM2 == GPU_R(0.0))
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
    return conductanceIntegral;
}

}
}

#undef GPU_WALL_INTERFACE_HD

#endif
