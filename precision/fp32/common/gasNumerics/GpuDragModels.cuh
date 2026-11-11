#ifndef UGKP_GPU_DRAG_MODELS_CUH
#define UGKP_GPU_DRAG_MODELS_CUH

#include "GpuDragAlgebra.cuh"

namespace ugkwpGpuDrag
{

                                                                            
                                                                            
                                                                         
struct DragInput
{
    GpuReal gasDensity;
    GpuReal gasViscosity;
    GpuReal gasVolumeFraction;
    GpuReal solidDensity;
    GpuReal diameter;
    GpuReal relativeSpeed;
    GpuReal denominatorRegularization;
    GpuReal parameter0;
    GpuReal parameter1;
    GpuReal parameter2;
    GpuReal parameter3;
};

struct SchillerNaumannDeviceDrag
{
    __device__ __forceinline__ static GpuReal reynoldsNumber
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

    __device__ __forceinline__ static GpuReal dragCoefficient
    (
        const GpuReal reynolds
    )
    {
        return
            ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannCoefficient
            (
                reynolds
            );
    }

    __device__ __forceinline__ static GpuReal inverseResponseTime
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
    __device__ __forceinline__ static GpuReal reynoldsNumber
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

    __device__ __forceinline__ static GpuReal cdRe
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

    __device__ __forceinline__ static GpuReal inverseResponseTime
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
