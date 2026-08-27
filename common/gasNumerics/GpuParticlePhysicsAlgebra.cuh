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

UGKWP_PARTICLE_PHYSICS_HD double ranzMarshallNuFromPr
(
    const double nonnegativeReynolds,
    const double positivePrandtl
)
{
    return
        2.0
      + 0.6*sqrt(nonnegativeReynolds)
         *pow(positivePrandtl, 1.0/3.0);
}

UGKWP_PARTICLE_PHYSICS_HD double ranzMarshallNuFromPrOneThird
(
    const double nonnegativeReynolds,
    const double prandtlOneThird
)
{
    return
        2.0 + 0.6*sqrt(nonnegativeReynolds)*prandtlOneThird;
}

UGKWP_PARTICLE_PHYSICS_HD double radialDistributionG0FromRatio
(
    const double concentrationRatio
)
{
    return
        (2.0 - concentrationRatio)
       /(2.0*pow(1.0 - concentrationRatio, 3.0) + 1.0e-5);
}

UGKWP_PARTICLE_PHYSICS_HD double collisionalPressure
(
    const double restitution,
    const double solidDensity,
    const double solidVolumeFraction,
    const double radialDistribution,
    const double granularTemperature
)
{
    return
        2.0*(1.0 + restitution)
       *solidDensity*solidVolumeFraction*solidVolumeFraction
       *radialDistribution*granularTemperature;
}

UGKWP_PARTICLE_PHYSICS_HD double granularMeanFreePath
(
    const double pi,
    const double diameter,
    const double solidVolumeFraction,
    const double radialDistribution,
    const double small
)
{
    return
        sqrt(pi)*diameter
       /(12.0*solidVolumeFraction*radialDistribution + small);
}

UGKWP_PARTICLE_PHYSICS_HD double granularCollisionTime
(
    const double meanFreePath,
    const double granularTemperature,
    const double small
)
{
    return meanFreePath/(sqrt(granularTemperature) + small);
}

}

#undef UGKWP_PARTICLE_PHYSICS_HD

#endif
