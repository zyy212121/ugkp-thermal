#ifndef UGKP_GPU_DRAG_MODELS_CUH
#define UGKP_GPU_DRAG_MODELS_CUH

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
        return
            input.gasDensity*input.diameter*input.relativeSpeed
           /fmax(input.gasViscosity, 1.0e-30);
    }

    __device__ __forceinline__ static double dragCoefficient
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

    __device__ __forceinline__ static double inverseResponseTime
    (
        const DragInput& input
    )
    {
        const double cd = dragCoefficient(reynoldsNumber(input));
        return
            0.75*cd*input.gasDensity*input.relativeSpeed
           /(
                input.solidDensity*input.diameter
              + input.denominatorRegularization
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
        return
            input.gasDensity*input.diameter*input.relativeSpeed
           /fmax(input.gasViscosity, 1.0e-30);
    }

    __device__ __forceinline__ static double cdRe
    (
        const DragInput& input
    )
    {
        const double alphaGas =
            fmin(fmax(input.gasVolumeFraction, 1.0e-12), 1.0);
        const double reynolds = fmax(reynoldsNumber(input), 0.0);

        if (alphaGas >= 0.8)
        {
            const double dispersedReynolds = alphaGas*reynolds;
            const double cdsReynolds = dispersedReynolds < 1000.0
              ? 24.0*(1.0 + 0.15*pow(dispersedReynolds, 0.687))
              : 0.44*fmax(dispersedReynolds, input.parameter0);
            return cdsReynolds*pow(alphaGas, -2.65);
        }

        return
            (4.0/3.0)
           *(
                150.0*(1.0 - alphaGas)/alphaGas
              + 1.75*reynolds
            );
    }

    __device__ __forceinline__ static double inverseResponseTime
    (
        const DragInput& input
    )
    {
        const double diameter = fmax(input.diameter, 1.0e-30);
        return
            0.75*cdRe(input)*fmax(input.gasViscosity, 1.0e-30)
           /(
                input.solidDensity*diameter*diameter
              + input.denominatorRegularization
            );
    }
};


struct ConstantResponseTimeDeviceDrag
{
    __device__ __forceinline__ static double reynoldsNumber
    (
        const DragInput& input
    )
    {
        return
            input.gasDensity*input.diameter*input.relativeSpeed
           /fmax(input.gasViscosity, 1.0e-30);
    }

    __device__ __forceinline__ static double inverseResponseTime
    (
        const DragInput& input
    )
    {
        return 1.0/fmax(input.parameter0, 1.0e-30);
    }
};

}                          

#endif
