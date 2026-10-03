#include "GpuPrecisionTypes.H"
#ifndef UGKWP_GPU_DRAG_ALGEBRA_CUH
#define UGKWP_GPU_DRAG_ALGEBRA_CUH

#include <cmath>

#if defined(__CUDACC__)
#define UGKWP_DRAG_HD __host__ __device__ __forceinline__
#else
#define UGKWP_DRAG_HD inline
#endif

namespace ugkwpGpuDragAlgebra
{

UGKWP_DRAG_HD GpuReal schillerNaumannCorrection(const GpuReal reynolds)
{
    return GPU_R(1.0) + GPU_R(0.15)*pow(reynolds, GPU_R(0.687));
}
UGKWP_DRAG_HD GpuReal gidaspowCdRe
(const GpuReal alpha, const GpuReal reynolds, const GpuReal residualRe)
{
    if (alpha >= GPU_R(0.8))
    {
        const GpuReal alphaRe = alpha*reynolds;
        const GpuReal cdRe = alphaRe < GPU_R(1000.0)
            ? GPU_R(24.0)*schillerNaumannCorrection(alphaRe)
            : GPU_R(0.44)*fmax(alphaRe, residualRe);
        return cdRe*pow(alpha, -GPU_R(2.65));
    }
    return (GPU_R(4.0)/GPU_R(3.0))
        *(GPU_R(150.0)*(GPU_R(1.0)-alpha)/alpha + GPU_R(1.75)*reynolds);
}

UGKWP_DRAG_HD GpuReal gasUgkpReynolds
(
    const GpuReal gasDensity,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal gasViscosity
)
{
    return
        gasDensity*diameter*relativeSpeed
       /fmax(gasViscosity, GPU_R(1.0e-30));
}

UGKWP_DRAG_HD GpuReal gasUgkpSchillerNaumannCoefficient
(
    const GpuReal reynolds
)
{
    const GpuReal reSafe = fmax(reynolds, GPU_R(1.0e-12));
    if (reSafe < GPU_R(1000.0))
    {
        return GPU_R(24.0)/reSafe*schillerNaumannCorrection(reSafe);
    }
    return GPU_R(0.44);
}

// Original gas inputs: retain caller-selected additive denominator protection.
struct RegularizedDragInputs
{
    static constexpr bool bounded = false;
    UGKWP_DRAG_HD static GpuReal density(GpuReal x) { return x; }
    UGKWP_DRAG_HD static GpuReal speed(GpuReal x) { return x; }
    UGKWP_DRAG_HD static GpuReal diameter(GpuReal x) { return x; }
    UGKWP_DRAG_HD static GpuReal coefficient(GpuReal re)
    { return gasUgkpSchillerNaumannCoefficient(re); }
    UGKWP_DRAG_HD static GpuReal denominator
    (GpuReal rhoS, GpuReal d, GpuReal regularization, bool squared)
    { return squared ? rhoS*d*d + regularization : rhoS*d + regularization; }
};

// Thermal inputs: preserve positivity bounds and the zero-slip return.
struct BoundedDragInputs
{
    static constexpr bool bounded = true;
    UGKWP_DRAG_HD static GpuReal density(GpuReal x) { return fmax(x,GPU_R(0.0)); }
    UGKWP_DRAG_HD static GpuReal speed(GpuReal x) { return fmax(x,GPU_R(0.0)); }
    UGKWP_DRAG_HD static GpuReal diameter(GpuReal x) { return fmax(x,GPU_R(1.0e-30)); }
    UGKWP_DRAG_HD static GpuReal coefficient(GpuReal re)
    {
        // Keep multiply-before-divide rounding, unlike the regularized policy.
        return re < GPU_R(1000.0)
          ? GPU_R(24.0)*schillerNaumannCorrection(re)/re : GPU_R(0.44);
    }
    UGKWP_DRAG_HD static GpuReal denominator
    (GpuReal rhoS, GpuReal d, GpuReal, bool squared)
    { return squared ? fmax(rhoS,GPU_R(1.0e-30))*d*d : fmax(rhoS,GPU_R(1.0e-30))*d; }
};

template<class InputPolicy>
UGKWP_DRAG_HD GpuReal inverseSchillerNaumannTime
(
    GpuReal gasDensity, GpuReal gasViscosity, GpuReal solidDensity,
    GpuReal diameter, GpuReal relativeSpeed, GpuReal denominatorRegularization
)
{
    const GpuReal mu = fmax(gasViscosity,GPU_R(1.0e-30));
    const GpuReal re = InputPolicy::density(gasDensity)
      *InputPolicy::diameter(diameter)*InputPolicy::speed(relativeSpeed)/mu;
    if constexpr (InputPolicy::bounded)
    {
        if (re <= GPU_R(1.0e-30) || relativeSpeed <= GPU_R(1.0e-30)) return GPU_R(0.0);
    }
    const GpuReal coefficient = InputPolicy::coefficient(re);
    return GPU_R(0.75)*coefficient*InputPolicy::density(gasDensity)
      *InputPolicy::speed(relativeSpeed)
      /InputPolicy::denominator(solidDensity,InputPolicy::diameter(diameter),denominatorRegularization,false);
}

template<class InputPolicy>
UGKWP_DRAG_HD GpuReal inverseGidaspowTime
(
    GpuReal gasDensity, GpuReal gasViscosity, GpuReal gasVolumeFraction,
    GpuReal solidDensity, GpuReal diameterInput, GpuReal relativeSpeed,
    GpuReal denominatorRegularization, GpuReal residualRe
)
{
    const GpuReal alpha = fmin(fmax(gasVolumeFraction,GPU_R(1.0e-12)),GPU_R(1.0));
    const GpuReal mu = fmax(gasViscosity,GPU_R(1.0e-30));
    const GpuReal diameter = fmax(diameterInput,GPU_R(1.0e-30));
    GpuReal re = InputPolicy::density(gasDensity)*InputPolicy::diameter(diameterInput)
      *InputPolicy::speed(relativeSpeed)/mu;
    if constexpr (!InputPolicy::bounded) re = fmax(re,GPU_R(0.0));
    const GpuReal cdRe = gidaspowCdRe(alpha,re,residualRe);
    return GPU_R(0.75)*cdRe*mu
      /InputPolicy::denominator(solidDensity,diameter,denominatorRegularization,true);
}

UGKWP_DRAG_HD GpuReal gasUgkpSchillerNaumannInverseResponseTime
(GpuReal rho, GpuReal mu, GpuReal rhoS, GpuReal d, GpuReal slip, GpuReal regularization)
{ return inverseSchillerNaumannTime<RegularizedDragInputs>(rho,mu,rhoS,d,slip,regularization); }

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowCdRe
(GpuReal rho, GpuReal mu, GpuReal alpha, GpuReal d, GpuReal slip, GpuReal residualRe)
{
    return gidaspowCdRe(fmin(fmax(alpha,GPU_R(1.0e-12)),GPU_R(1.0)),
        fmax(gasUgkpReynolds(rho,d,slip,mu),GPU_R(0.0)),residualRe);
}

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowInverseResponseTime
(GpuReal rho, GpuReal mu, GpuReal alpha, GpuReal rhoS, GpuReal d, GpuReal slip,
 GpuReal regularization, GpuReal residualRe)
{ return inverseGidaspowTime<RegularizedDragInputs>(rho,mu,alpha,rhoS,d,slip,regularization,residualRe); }

UGKWP_DRAG_HD GpuReal fshChtSchillerNaumannInverseRelaxationTime
(GpuReal rho, GpuReal mu, GpuReal rhoS, GpuReal d, GpuReal slip)
{ return inverseSchillerNaumannTime<BoundedDragInputs>(rho,mu,rhoS,d,slip,GPU_R(0.0)); }

UGKWP_DRAG_HD GpuReal fshChtGidaspowInverseRelaxationTime
(GpuReal rho, GpuReal alpha, GpuReal mu, GpuReal rhoS, GpuReal d, GpuReal slip, GpuReal residualRe)
{ return inverseGidaspowTime<BoundedDragInputs>(rho,mu,alpha,rhoS,d,slip,GPU_R(0.0),residualRe); }

}

#undef UGKWP_DRAG_HD
#endif
