#pragma once
__device__ inline GpuReal coldWall1DGroupSum
(
    GpuReal value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

__device__ inline GpuReal coldWall1DGroupMinPrefix
(
    GpuReal value,
    const int lane,
    const unsigned int mask
)
{
    for (int offset = 1; offset < 8; offset *= 2)
    {
        const GpuReal lower = __shfl_up_sync(mask, value, offset, 8);
        if (lane >= offset && lower < value)
        {
            value = lower;
        }
    }
    return value;
}

__device__ inline GpuReal coldWall1DPcrSolve
(
    GpuReal lower,
    GpuReal diagonal,
    GpuReal upper,
    GpuReal rightHandSide,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const GpuReal lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const GpuReal lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const GpuReal lowerUpper =
            __shfl_up_sync(mask, upper, stride, 8);
        const GpuReal lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const GpuReal upperLower =
            __shfl_down_sync(mask, lower, stride, 8);
        const GpuReal upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const GpuReal upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const GpuReal upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const GpuReal alpha = lane >= stride
          ? -lower/(lowerDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal beta = lane + stride < 8
          ? -upper/(upperDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const GpuReal nextLower = lane >= stride
          ? alpha*lowerLower
          : GPU_R(0.0);
        const GpuReal nextUpper = lane + stride < 8
          ? beta*upperUpper
          : GPU_R(0.0);
        const GpuReal nextDiagonal = diagonal
          + alpha*lowerUpper + beta*upperLower;
        const GpuReal nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + GPU_REAL_MIN);
}

__device__ inline GpuReal coldWall1DSolidFractionFromKnownTemperature
(
    const GpuReal temperatureK,
    const Foam::gpuThermal::ColdWallSolidificationParameters& parameters
)
{
    const GpuReal solidus = Foam::gpuThermal::coldWallSolidusTemperature(parameters);
    const GpuReal liquidus = Foam::gpuThermal::coldWallLiquidusTemperature(parameters);
    if (temperatureK <= solidus) return GPU_R(1.0);
    if (temperatureK >= liquidus) return GPU_R(0.0);
    return (liquidus - temperatureK)/parameters.mushyRangeK;
}
