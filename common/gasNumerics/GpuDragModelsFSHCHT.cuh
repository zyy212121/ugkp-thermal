#include "GpuPrecisionTypes.H"
#ifndef UGKWP_GPU_DRAG_MODELS_CUH
#define UGKWP_GPU_DRAG_MODELS_CUH

#include "GpuDragAlgebra.cuh"

namespace ugkwpGpuDrag
{

struct DragInput
{
    GpuReal gasDensity;
    GpuReal gasVolumeFraction;
    GpuReal gasViscosity;
    GpuReal solidDensity;
    GpuReal diameter;
    GpuReal relativeSpeed;
};

struct SchillerNaumannDrag
{
    __device__ static GpuReal inverseRelaxationTime(const DragInput& in)
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
    GpuReal residualRe;

    __device__ GpuReal inverseRelaxationTime(const DragInput& in) const
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
