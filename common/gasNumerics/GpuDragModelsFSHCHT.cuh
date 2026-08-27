#ifndef UGKWP_GPU_DRAG_MODELS_CUH
#define UGKWP_GPU_DRAG_MODELS_CUH

#include "GpuDragAlgebra.cuh"

namespace ugkwpGpuDrag
{

struct DragInput
{
    double gasDensity;
    double gasVolumeFraction;
    double gasViscosity;
    double solidDensity;
    double diameter;
    double relativeSpeed;
};

struct SchillerNaumannDrag
{
    __device__ static double inverseRelaxationTime(const DragInput& in)
    {
        return
            ugkwpGpuDragAlgebra::fshChtSchillerNaumannInverseRelaxationTime
            (
                in.gasDensity,
                in.gasViscosity,
                in.solidDensity,
                in.diameter,
                in.relativeSpeed
            );
    }
};

struct GidaspowErgunWenYuDrag
{
    double residualRe;

    __device__ double inverseRelaxationTime(const DragInput& in) const
    {
        return
            ugkwpGpuDragAlgebra::fshChtGidaspowInverseRelaxationTime
            (
                in.gasDensity,
                in.gasVolumeFraction,
                in.gasViscosity,
                in.solidDensity,
                in.diameter,
                in.relativeSpeed,
                residualRe
            );
    }
};

}                          

#endif
