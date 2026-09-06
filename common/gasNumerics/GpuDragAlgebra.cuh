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
        return GPU_R(24.0)/reSafe*(GPU_R(1.0) + GPU_R(0.15)*pow(reSafe, GPU_R(0.687)));
    }
    return GPU_R(0.44);
}

UGKWP_DRAG_HD GpuReal gasUgkpSchillerNaumannInverseResponseTime
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal solidDensity,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal denominatorRegularization
)
{
    const GpuReal reynolds = gasUgkpReynolds
    (
        gasDensity,
        diameter,
        relativeSpeed,
        gasViscosity
    );
    const GpuReal coefficient =
        gasUgkpSchillerNaumannCoefficient(reynolds);
    return
        GPU_R(0.75)*coefficient*gasDensity*relativeSpeed
       /(solidDensity*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowCdRe
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal gasVolumeFraction,
    const GpuReal diameter,
    const GpuReal relativeSpeed,
    const GpuReal residualRe
)
{
    const GpuReal alphaGas =
        fmin(fmax(gasVolumeFraction, GPU_R(1.0e-12)), GPU_R(1.0));
    const GpuReal reynolds = fmax
    (
        gasUgkpReynolds
        (
            gasDensity,
            diameter,
            relativeSpeed,
            gasViscosity
        ),
        GPU_R(0.0)
    );
    if (alphaGas >= GPU_R(0.8))
    {
        const GpuReal dispersedReynolds = alphaGas*reynolds;
        const GpuReal cdsReynolds = dispersedReynolds < GPU_R(1000.0)
          ? GPU_R(24.0)*(GPU_R(1.0) + GPU_R(0.15)*pow(dispersedReynolds, GPU_R(0.687)))
          : GPU_R(0.44)*fmax(dispersedReynolds, residualRe);
        return cdsReynolds*pow(alphaGas, -GPU_R(2.65));
    }
    return
        (GPU_R(4.0)/GPU_R(3.0))
       *(
            GPU_R(150.0)*(GPU_R(1.0) - alphaGas)/alphaGas
          + GPU_R(1.75)*reynolds
        );
}

UGKWP_DRAG_HD GpuReal gasUgkpGidaspowInverseResponseTime
(
    const GpuReal gasDensity,
    const GpuReal gasViscosity,
    const GpuReal gasVolumeFraction,
    const GpuReal solidDensity,
    const GpuReal diameterInput,
    const GpuReal relativeSpeed,
    const GpuReal denominatorRegularization,
    const GpuReal residualRe
)
{
    const GpuReal diameter = fmax(diameterInput, GPU_R(1.0e-30));
    return
        GPU_R(0.75)
       *gasUgkpGidaspowCdRe
        (
            gasDensity,
            gasViscosity,
            gasVolumeFraction,
            diameterInput,
            relativeSpeed,
            residualRe
        )
       *fmax(gasViscosity, GPU_R(1.0e-30))
       /(solidDensity*diameter*diameter + denominatorRegularization);
}

UGKWP_DRAG_HD GpuReal fshChtSchillerNaumannInverseRelaxationTime
(
    const GpuReal gasDensityInput,
    const GpuReal gasViscosity,
    const GpuReal solidDensityInput,
    const GpuReal diameterInput,
    const GpuReal relativeSpeedInput
)
{
    const GpuReal mu = fmax(gasViscosity, GPU_R(1.0e-30));
    const GpuReal re =
        fmax(gasDensityInput, GPU_R(0.0))*fmax(diameterInput, GPU_R(1.0e-30))
       *fmax(relativeSpeedInput, GPU_R(0.0))/mu;
    if (re <= GPU_R(1.0e-30) || relativeSpeedInput <= GPU_R(1.0e-30))
    {
        return GPU_R(0.0);
    }
    const GpuReal coefficient =
        re < GPU_R(1000.0)
      ? GPU_R(24.0)*(GPU_R(1.0) + GPU_R(0.15)*pow(re, GPU_R(0.687)))/re
      : GPU_R(0.44);
    return
        GPU_R(0.75)*coefficient*fmax(gasDensityInput, GPU_R(0.0))
       *fmax(relativeSpeedInput, GPU_R(0.0))
       /(fmax(solidDensityInput, GPU_R(1.0e-30))
        *fmax(diameterInput, GPU_R(1.0e-30)));
}

UGKWP_DRAG_HD GpuReal fshChtGidaspowInverseRelaxationTime
(
    const GpuReal gasDensityInput,
    const GpuReal gasVolumeFraction,
    const GpuReal gasViscosity,
    const GpuReal solidDensityInput,
    const GpuReal diameterInput,
    const GpuReal relativeSpeedInput,
    const GpuReal residualRe
)
{
    const GpuReal alpha =
        fmin(fmax(gasVolumeFraction, GPU_R(1.0e-12)), GPU_R(1.0));
    const GpuReal mu = fmax(gasViscosity, GPU_R(1.0e-30));
    const GpuReal diameter = fmax(diameterInput, GPU_R(1.0e-30));
    const GpuReal re =
        fmax(gasDensityInput, GPU_R(0.0))*diameter
       *fmax(relativeSpeedInput, GPU_R(0.0))/mu;
    const GpuReal alphaRe = alpha*re;
    const GpuReal cdReWenYu =
        alphaRe < GPU_R(1000.0)
      ? GPU_R(24.0)*(GPU_R(1.0) + GPU_R(0.15)*pow(fmax(alphaRe, GPU_R(0.0)), GPU_R(0.687)))
      : GPU_R(0.44)*fmax(alphaRe, residualRe);
    const GpuReal cdRe =
        alpha >= GPU_R(0.8)
      ? cdReWenYu*pow(alpha, -GPU_R(2.65))
      : (GPU_R(4.0)/GPU_R(3.0))
       *(GPU_R(150.0)*(GPU_R(1.0) - alpha)/alpha + GPU_R(1.75)*re);
    return
        GPU_R(0.75)*cdRe*mu
       /(fmax(solidDensityInput, GPU_R(1.0e-30))*diameter*diameter);
}

}

#undef UGKWP_DRAG_HD

#endif
