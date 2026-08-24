#ifndef UGKWP_GPU_DRAG_MODEL_H
#define UGKWP_GPU_DRAG_MODEL_H

#include "dictionary.H"
#include "error.H"
#include "word.H"

#include <cmath>

namespace Foam
{

enum class GpuDragModelId : int
{
    none = 0,
    SchillerNaumann = 1,
    GidaspowErgunWenYu = 2
};

struct GpuDragModelSelection
{
    GpuDragModelId id;
    word name;
    scalar residualRe;
};

inline GpuDragModelSelection selectGpuDragModel(const dictionary& dict)
{
    const word name
    (
        dict.lookupOrDefault<word>("dragModel", "SchillerNaumann")
    );
    if (name == "none")
    {
        return
        {
            GpuDragModelId::none,
            name,
            scalar(1.0e-3)
        };
    }
    if (name == "SchillerNaumann")
    {
        return
        {
            GpuDragModelId::SchillerNaumann,
            name,
            scalar(1.0e-3)
        };
    }
    if (name == "GidaspowErgunWenYu")
    {
        if (!dict.found("GidaspowErgunWenYuCoeffs"))
        {
            FatalErrorInFunction
                << "dragModel GidaspowErgunWenYu requires the "
                << "GidaspowErgunWenYuCoeffs dictionary."
                << exit(FatalError);
        }
        const dictionary& coeffs =
            dict.subDict("GidaspowErgunWenYuCoeffs");
        const scalar residualRe =
            coeffs.lookupOrDefault<scalar>("residualRe", 1.0e-3);
        if
        (
            !std::isfinite(static_cast<double>(residualRe))
         || residualRe <= SMALL
        )
        {
            FatalErrorInFunction
                << "GidaspowErgunWenYuCoeffs/residualRe must be finite "
                << "and positive. Current value = " << residualRe
                << exit(FatalError);
        }
        return
        {
            GpuDragModelId::GidaspowErgunWenYu,
            name,
            residualRe
        };
    }

    FatalErrorInFunction
        << "Unknown dragModel '" << name << "'. Supported models are "
        << "none, SchillerNaumann and GidaspowErgunWenYu."
        << exit(FatalError);
    return
    {
        GpuDragModelId::SchillerNaumann,
        "SchillerNaumann",
        scalar(1.0e-3)
    };
}

}                  

#endif
