#include "GpuPrecisionTypes.H"
#ifndef GPU_COUPLING_MATH_H
#define GPU_COUPLING_MATH_H

#include <cmath>

#if defined(__CUDACC__)
#define GPU_COUPLING_HD __host__ __device__
#else
#define GPU_COUPLING_HD
#endif

template<class Real> GPU_COUPLING_HD inline Real gpuCouplingOneMinusExpNegCore(const Real x)
{
    if (!(x > Real(0.0))) return Real(0.0);
    if (x < Real(1.0e-6)) return x*(Real(1.0)-Real(0.5)*x+x*x/Real(6.0));
    return Real(1.0)-::exp(-x);
}
GPU_COUPLING_HD inline GpuReal gpuCouplingOneMinusExpNeg(const GpuReal x)
{return gpuCouplingOneMinusExpNegCore(x);}
template<class Real,class Time> GPU_COUPLING_HD inline Real gpuFiniteCapacityHeatExchangeCore
(
    const Real gasCapacity,const Real solidCapacity,const Real gasTemperature,
    const Real solidTemperature,const Real solidRelaxationRate,const Time dt
)
{
    if (!(gasCapacity>Real(0.0))||!(solidCapacity>Real(0.0))
        ||!(solidRelaxationRate>Real(0.0))||!(dt>Time(0.0))) return Real(0.0);
    const Real totalCapacity=gasCapacity+solidCapacity;
    const Real effectiveCapacity=gasCapacity*solidCapacity/totalCapacity;
    const Real coupledRate=solidRelaxationRate*(Real(1.0)+solidCapacity/gasCapacity);
    return effectiveCapacity*(gasTemperature-solidTemperature)
        *gpuCouplingOneMinusExpNegCore(coupledRate*Real(dt));
}
GPU_COUPLING_HD inline GpuReal gpuFiniteCapacityHeatExchange
(
    const GpuReal gasCapacity,const GpuReal solidCapacity,const GpuReal gasTemperature,
    const GpuReal solidTemperature,const GpuReal solidRelaxationRate,const GpuTime dt
)
{return gpuFiniteCapacityHeatExchangeCore(gasCapacity,solidCapacity,gasTemperature,
    solidTemperature,solidRelaxationRate,dt);}

#undef GPU_COUPLING_HD

#endif
