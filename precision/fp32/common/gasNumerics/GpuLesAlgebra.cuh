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

// These invariants are formed from a gradient divided by its largest
// component. Their high powers are dimensionless, including diagnostics.
struct WaleTerms
{
    GpuReal numerator;
    GpuReal denominator;
};

UGKWP_LES_HD WaleTerms waleTerms
(
    const GpuReal symmetricGradientSquared,
    const GpuReal tracelessSquaredGradientSquared
)
{
    const GpuReal s2 = fmaxf(symmetricGradientSquared, GPU_R(0.0));
    const GpuReal sd2 = fmaxf(tracelessSquaredGradientSquared, GPU_R(0.0));
    return {powf(sd2, GPU_R(1.5)),
        powf(s2, GPU_R(2.5)) + powf(sd2, GPU_R(1.25))};
}

UGKWP_LES_HD GpuReal waleNutFromTerms
(
    const GpuReal coefficient,
    const GpuReal delta,
    const GpuReal gradientScale,
    const WaleTerms terms,
    const GpuReal small
)
{
    if (gradientScale == GPU_R(0.0)) return GPU_R(0.0);
    // Preserve the original dimensional denominator > small criterion.
    // Divide sequentially instead of forming gradientScale to the fifth.
    GpuReal scaledSmall = small;
    for (int i = 0; i < 5; ++i) scaledSmall /= gradientScale;
    if (!(terms.denominator > scaledSmall)) return GPU_R(0.0);

    // Restore nu = (Cw*delta)^2 * gradientScale * numerator/denominator.
    // Keep binary exponents separate so a representable final viscosity
    // does not disappear/overflow while squaring delta or restoring units.
    int ce = 0, de = 0, ge = 0, re = 0;
    const GpuReal cm = frexpf(coefficient, &ce);
    const GpuReal dm = frexpf(delta, &de);
    const GpuReal gm = frexpf(gradientScale, &ge);
    const GpuReal rm = frexpf(terms.numerator/terms.denominator, &re);
    return ldexpf((cm*cm)*(dm*dm)*(gm*rm), 2*ce + 2*de + ge + re);
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
    // Retain the invariant-based entry for callers that already have them.
    // The production kernel scales before even forming these invariants.
    const GpuReal s2 = fmaxf(symmetricGradientSquared, GPU_R(0.0));
    const GpuReal sd2 = fmaxf(tracelessSquaredGradientSquared, GPU_R(0.0));
    const GpuReal gradientScale = fmaxf(sqrtf(s2), sqrtf(sqrtf(sd2)));
    if (gradientScale == GPU_R(0.0)) return GPU_R(0.0);
    const GpuReal scaledS2 = (s2/gradientScale)/gradientScale;
    const GpuReal scaledSd2 = ((sd2/gradientScale)/gradientScale)
        /gradientScale/gradientScale;
    return waleNutFromTerms(coefficient, delta, gradientScale,
        waleTerms(scaledS2, scaledSd2), small);
}

}

#undef UGKWP_LES_HD

#endif
