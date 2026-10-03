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
    return gidaspowCdRe(alphaGas, reynolds, residualRe);
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
      ? GPU_R(24.0)*schillerNaumannCorrection(re)/re
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
    const GpuReal cdRe = gidaspowCdRe(alpha, re, residualRe);
    return
        GPU_R(0.75)*cdRe*mu
       /(fmax(solidDensityInput, GPU_R(1.0e-30))*diameter*diameter);
}

}

#undef UGKWP_DRAG_HD

#endif
