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

UGKWP_DRAG_HD double gasUgkpReynolds
(
    const double gasDensity,
    const double diameter,
    const double relativeSpeed,
    const double gasViscosity
)
{
    return
        gasDensity*diameter*relativeSpeed
       /fmax(gasViscosity, 1.0e-30);
}

UGKWP_DRAG_HD double gasUgkpSchillerNaumannCoefficient
(
    const double reynolds
)
{
    const double reSafe = fmax(reynolds, 1.0e-12);
    if (reSafe < 1000.0)
    {
        return 24.0/reSafe*(1.0 + 0.15*pow(reSafe, 0.687));
    }
    return 0.44;
}

UGKWP_DRAG_HD double gasUgkpSchillerNaumannInverseResponseTime
(
    const double gasDensity,
    const double gasViscosity,
    const double solidDensity,
    const double diameter,
    const double relativeSpeed,
    const double denominatorRegularization
)
{
    const double reynolds = gasUgkpReynolds
    (
        gasDensity,
        diameter,
        relativeSpeed,
        gasViscosity
    );
    const double coefficient =
        gasUgkpSchillerNaumannCoefficient(reynolds);
    return
        0.75*coefficient*gasDensity*relativeSpeed
       /(solidDensity*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD double gasUgkpGidaspowCdRe
(
    const double gasDensity,
    const double gasViscosity,
    const double gasVolumeFraction,
    const double diameter,
    const double relativeSpeed,
    const double residualRe
)
{
    const double alphaGas =
        fmin(fmax(gasVolumeFraction, 1.0e-12), 1.0);
    const double reynolds = fmax
    (
        gasUgkpReynolds
        (
            gasDensity,
            diameter,
            relativeSpeed,
            gasViscosity
        ),
        0.0
    );
    if (alphaGas >= 0.8)
    {
        const double dispersedReynolds = alphaGas*reynolds;
        const double cdsReynolds = dispersedReynolds < 1000.0
          ? 24.0*(1.0 + 0.15*pow(dispersedReynolds, 0.687))
          : 0.44*fmax(dispersedReynolds, residualRe);
        return cdsReynolds*pow(alphaGas, -2.65);
    }
    return
        (4.0/3.0)
       *(
            150.0*(1.0 - alphaGas)/alphaGas
          + 1.75*reynolds
        );
}

UGKWP_DRAG_HD double gasUgkpGidaspowInverseResponseTime
(
    const double gasDensity,
    const double gasViscosity,
    const double gasVolumeFraction,
    const double solidDensity,
    const double diameterInput,
    const double relativeSpeed,
    const double denominatorRegularization,
    const double residualRe
)
{
    const double diameter = fmax(diameterInput, 1.0e-30);
    return
        0.75
       *gasUgkpGidaspowCdRe
        (
            gasDensity,
            gasViscosity,
            gasVolumeFraction,
            diameterInput,
            relativeSpeed,
            residualRe
        )
       *fmax(gasViscosity, 1.0e-30)
       /(solidDensity*diameter*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD double fshChtSchillerNaumannInverseRelaxationTime
(
    const double gasDensityInput,
    const double gasViscosity,
    const double solidDensityInput,
    const double diameterInput,
    const double relativeSpeedInput
)
{
    const double mu = fmax(gasViscosity, 1.0e-30);
    const double re =
        fmax(gasDensityInput, 0.0)*fmax(diameterInput, 1.0e-30)
       *fmax(relativeSpeedInput, 0.0)/mu;
    if (re <= 1.0e-30 || relativeSpeedInput <= 1.0e-30)
    {
        return 0.0;
    }
    const double coefficient =
        re < 1000.0
      ? 24.0*(1.0 + 0.15*pow(re, 0.687))/re
      : 0.44;
    return
        0.75*coefficient*fmax(gasDensityInput, 0.0)
       *fmax(relativeSpeedInput, 0.0)
       /(fmax(solidDensityInput, 1.0e-30)
        *fmax(diameterInput, 1.0e-30));
}

UGKWP_DRAG_HD double fshChtGidaspowInverseRelaxationTime
(
    const double gasDensityInput,
    const double gasVolumeFraction,
    const double gasViscosity,
    const double solidDensityInput,
    const double diameterInput,
    const double relativeSpeedInput,
    const double residualRe
)
{
    const double alpha =
        fmin(fmax(gasVolumeFraction, 1.0e-12), 1.0);
    const double mu = fmax(gasViscosity, 1.0e-30);
    const double diameter = fmax(diameterInput, 1.0e-30);
    const double re =
        fmax(gasDensityInput, 0.0)*diameter
       *fmax(relativeSpeedInput, 0.0)/mu;
    const double alphaRe = alpha*re;
    const double cdReWenYu =
        alphaRe < 1000.0
      ? 24.0*(1.0 + 0.15*pow(fmax(alphaRe, 0.0), 0.687))
      : 0.44*fmax(alphaRe, residualRe);
    const double cdRe =
        alpha >= 0.8
      ? cdReWenYu*pow(alpha, -2.65)
      : (4.0/3.0)
       *(150.0*(1.0 - alpha)/alpha + 1.75*re);
    return
        0.75*cdRe*mu
       /(fmax(solidDensityInput, 1.0e-30)*diameter*diameter);
}

}

#undef UGKWP_DRAG_HD

#endif
