#include "GpuPrecisionTypes.H"
#ifndef GPU_COUPLING_MATH_H
#define GPU_COUPLING_MATH_H

#include <cmath>

#if defined(__CUDACC__)
#define GPU_COUPLING_HD __host__ __device__
#else
#define GPU_COUPLING_HD
#endif

GPU_COUPLING_HD inline GpuEnergy gpuCouplingOneMinusExpNeg(const GpuEnergy x)
{
    if (!(x > GPU_R(0.0)))
    {
        return GPU_R(0.0);
    }

    if (x < GPU_R(1.0e-6))
    {
        return x*(GPU_R(1.0) - GPU_R(0.5)*x + x*x/GPU_R(6.0));
    }

    return GPU_R(1.0) - ::exp(-x);
}

GPU_COUPLING_HD inline GpuEnergy gpuFiniteCapacityHeatExchange
(
    const GpuEnergy gasCapacity,
    const GpuEnergy solidCapacity,
    const GpuEnergy gasTemperature,
    const GpuEnergy solidTemperature,
    const GpuEnergy solidRelaxationRate,
    const GpuTime dt
)
{
    if
    (
        !(gasCapacity > GPU_R(0.0))
     || !(solidCapacity > GPU_R(0.0))
     || !(solidRelaxationRate > GPU_R(0.0))
     || !(dt > GPU_R(0.0))
    )
    {
        return GPU_R(0.0);
    }

    const GpuEnergy totalCapacity = gasCapacity + solidCapacity;
    const GpuEnergy effectiveCapacity =
        gasCapacity*solidCapacity/totalCapacity;
    const GpuEnergy coupledRate =
        solidRelaxationRate*(GPU_R(1.0) + solidCapacity/gasCapacity);

    return effectiveCapacity
       *(gasTemperature - solidTemperature)
       *gpuCouplingOneMinusExpNeg(coupledRate*dt);
}

#undef GPU_COUPLING_HD

#endif
