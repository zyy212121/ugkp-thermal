#ifndef GPU_COUPLING_MATH_H
#define GPU_COUPLING_MATH_H

#include <cmath>

#if defined(__CUDACC__)
#define GPU_COUPLING_HD __host__ __device__
#else
#define GPU_COUPLING_HD
#endif

GPU_COUPLING_HD inline double gpuCouplingOneMinusExpNeg(const double x)
{
    if (!(x > 0.0))
    {
        return 0.0;
    }

    if (x < 1.0e-6)
    {
        return x*(1.0 - 0.5*x + x*x/6.0);
    }

    return 1.0 - ::exp(-x);
}

GPU_COUPLING_HD inline double gpuFiniteCapacityHeatExchange
(
    const double gasCapacity,
    const double solidCapacity,
    const double gasTemperature,
    const double solidTemperature,
    const double solidRelaxationRate,
    const double dt
)
{
    if
    (
        !(gasCapacity > 0.0)
     || !(solidCapacity > 0.0)
     || !(solidRelaxationRate > 0.0)
     || !(dt > 0.0)
    )
    {
        return 0.0;
    }

    const double totalCapacity = gasCapacity + solidCapacity;
    const double effectiveCapacity =
        gasCapacity*solidCapacity/totalCapacity;
    const double coupledRate =
        solidRelaxationRate*(1.0 + solidCapacity/gasCapacity);

    return effectiveCapacity
       *(gasTemperature - solidTemperature)
       *gpuCouplingOneMinusExpNeg(coupledRate*dt);
}

#undef GPU_COUPLING_HD

#endif
