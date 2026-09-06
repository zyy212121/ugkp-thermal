#include "GpuPrecisionTypes.H"
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

UGKWP_LES_HD GpuReal smagorinskyNut
(
    const GpuReal coefficient,
    const GpuReal delta,
    const GpuReal deviatoricStrainSquared
)
{
    return
        coefficient*coefficient*delta*delta
       *sqrt(fmax(GPU_R(2.0)*deviatoricStrainSquared, GPU_R(0.0)));
}

UGKWP_LES_HD GpuReal waleNut
(
    const GpuReal coefficient,
    const GpuReal delta,
    const GpuReal symmetricGradientSquared,
    const GpuReal tracelessSquaredGradientSquared,
    const GpuReal small
)
{
    const GpuReal numerator =
        pow(fmax(tracelessSquaredGradientSquared, GPU_R(0.0)), GPU_R(1.5));
    const GpuReal denominator =
        pow(fmax(symmetricGradientSquared, GPU_R(0.0)), GPU_R(2.5))
      + pow(fmax(tracelessSquaredGradientSquared, GPU_R(0.0)), GPU_R(1.25));
    return denominator > small
      ? coefficient*coefficient*delta*delta*numerator/denominator
      : GPU_R(0.0);
}

}

#undef UGKWP_LES_HD

#endif
