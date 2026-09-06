#include "GpuPrecisionTypes.H"
#ifndef UGKWP_GPU_PARTICLE_PHYSICS_ALGEBRA_CUH
#define UGKWP_GPU_PARTICLE_PHYSICS_ALGEBRA_CUH

#include <cmath>

#if defined(__CUDACC__)
#define UGKWP_PARTICLE_PHYSICS_HD __host__ __device__ __forceinline__
#else
#define UGKWP_PARTICLE_PHYSICS_HD inline
#endif

namespace ugkwp
{

UGKWP_PARTICLE_PHYSICS_HD GpuReal ranzMarshallNuFromPr
(
    const GpuReal nonnegativeReynolds,
    const GpuReal positivePrandtl
)
{
    return
        GPU_R(2.0)
      + GPU_R(0.6)*sqrt(nonnegativeReynolds)
         *pow(positivePrandtl, GPU_R(1.0)/GPU_R(3.0));
}

UGKWP_PARTICLE_PHYSICS_HD GpuReal ranzMarshallNuFromPrOneThird
(
    const GpuReal nonnegativeReynolds,
    const GpuReal prandtlOneThird
)
{
    return
        GPU_R(2.0) + GPU_R(0.6)*sqrt(nonnegativeReynolds)*prandtlOneThird;
}

UGKWP_PARTICLE_PHYSICS_HD GpuReal radialDistributionG0FromRatio
(
    const GpuReal concentrationRatio
)
{
    return
        (GPU_R(2.0) - concentrationRatio)
       /(GPU_R(2.0)*pow(GPU_R(1.0) - concentrationRatio, GPU_R(3.0)) + GPU_R(1.0e-5));
}

UGKWP_PARTICLE_PHYSICS_HD GpuReal collisionalPressure
(
    const GpuReal restitution,
    const GpuReal solidDensity,
    const GpuReal solidVolumeFraction,
    const GpuReal radialDistribution,
    const GpuReal granularTemperature
)
{
    return
        GPU_R(2.0)*(GPU_R(1.0) + restitution)
       *solidDensity*solidVolumeFraction*solidVolumeFraction
       *radialDistribution*granularTemperature;
}

UGKWP_PARTICLE_PHYSICS_HD GpuReal granularMeanFreePath
(
    const GpuReal pi,
    const GpuReal diameter,
    const GpuReal solidVolumeFraction,
    const GpuReal radialDistribution,
    const GpuReal small
)
{
    return
        sqrt(pi)*diameter
       /(GPU_R(12.0)*solidVolumeFraction*radialDistribution + small);
}

UGKWP_PARTICLE_PHYSICS_HD GpuReal granularCollisionTime
(
    const GpuReal meanFreePath,
    const GpuReal granularTemperature,
    const GpuReal small
)
{
    return meanFreePath/(sqrt(granularTemperature) + small);
}

}

#undef UGKWP_PARTICLE_PHYSICS_HD

#endif
