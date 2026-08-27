#ifndef UGKWP_GPU_LES_ALGEBRA_CUH
#define UGKWP_GPU_LES_ALGEBRA_CUH

#include <cmath>

#if defined(__CUDACC__)
#define UGKWP_LES_HD __host__ __device__ __forceinline__
#else
#define UGKWP_LES_HD inline
#endif

namespace ugkwp
{

UGKWP_LES_HD double smagorinskyNut
(
    const double coefficient,
    const double delta,
    const double deviatoricStrainSquared
)
{
    return
        coefficient*coefficient*delta*delta
       *sqrt(fmax(2.0*deviatoricStrainSquared, 0.0));
}

UGKWP_LES_HD double waleNut
(
    const double coefficient,
    const double delta,
    const double symmetricGradientSquared,
    const double tracelessSquaredGradientSquared,
    const double small
)
{
    const double numerator =
        pow(fmax(tracelessSquaredGradientSquared, 0.0), 1.5);
    const double denominator =
        pow(fmax(symmetricGradientSquared, 0.0), 2.5)
      + pow(fmax(tracelessSquaredGradientSquared, 0.0), 1.25);
    return denominator > small
      ? coefficient*coefficient*delta*delta*numerator/denominator
      : 0.0;
}

}

#undef UGKWP_LES_HD

#endif
