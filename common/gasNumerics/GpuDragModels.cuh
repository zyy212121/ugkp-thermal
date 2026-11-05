#ifndef UGKP_GPU_DRAG_MODELS_CUH
#define UGKP_GPU_DRAG_MODELS_CUH

#include "GpuDragAlgebra.cuh"

namespace ugkwpGpuDrag
{

                                                                            
                                                                            
                                                                         
struct DragInput
{
    double gasDensity;
    double gasViscosity;
    double gasVolumeFraction;
    double solidDensity;
    double diameter;
    double relativeSpeed;
    double denominatorRegularization;
    double parameter0;
    double parameter1;
    double parameter2;
    double parameter3;
};

struct SchillerNaumannDeviceDrag
{
    __device__ __forceinline__ static double reynoldsNumber
    (
        const DragInput& input
    )
    {
        return ugkwpGpuDragAlgebra::gasUgkpReynolds
        (
            input.gasDensity,
            input.diameter,
            input.relativeSpeed,
            input.gasViscosity
        );
    }

    __device__ __forceinline__ static double dragCoefficient
    (
        const double reynolds
    )
    {
        return
            ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannCoefficient
            (
                reynolds
            );
    }

    __device__ __forceinline__ static double inverseResponseTime
    (
        const DragInput& input
    )
    {
        return
            ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannInverseResponseTime
            (
                input.gasDensity,
                input.gasViscosity,
                input.solidDensity,
                input.diameter,
                input.relativeSpeed,
                input.denominatorRegularization
            );
    }
};


struct GidaspowErgunWenYuDeviceDrag
{
    __device__ __forceinline__ static double reynoldsNumber
    (
        const DragInput& input
    )
    {
        return ugkwpGpuDragAlgebra::gasUgkpReynolds
        (
            input.gasDensity,
            input.diameter,
            input.relativeSpeed,
            input.gasViscosity
        );
    }

    __device__ __forceinline__ static double cdRe
    (
        const DragInput& input
    )
    {
        return ugkwpGpuDragAlgebra::gasUgkpGidaspowCdRe
        (
            input.gasDensity,
            input.gasViscosity,
            input.gasVolumeFraction,
            input.diameter,
            input.relativeSpeed,
            input.parameter0
        );
    }

    __device__ __forceinline__ static double inverseResponseTime
    (
        const DragInput& input
    )
    {
        return
            ugkwpGpuDragAlgebra::gasUgkpGidaspowInverseResponseTime
            (
                input.gasDensity,
                input.gasViscosity,
                input.gasVolumeFraction,
                input.solidDensity,
                input.diameter,
                input.relativeSpeed,
                input.denominatorRegularization,
                input.parameter0
            );
    }
};

}                          

#endif
