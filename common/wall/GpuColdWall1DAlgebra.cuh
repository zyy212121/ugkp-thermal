#pragma once
template<class Scalar>
__device__ inline Scalar coldWall1DGroupSum
(
    Scalar value,
    const unsigned int mask
)
{
    value += __shfl_down_sync(mask, value, 4, 8);
    value += __shfl_down_sync(mask, value, 2, 8);
    value += __shfl_down_sync(mask, value, 1, 8);
    return __shfl_sync(mask, value, 0, 8);
}

template<class Scalar>
__device__ inline Scalar coldWall1DGroupMinPrefix
(
    Scalar value,
    const int lane,
    const unsigned int mask
)
{
    for (int offset = 1; offset < 8; offset *= 2)
    {
        const Scalar lower = __shfl_up_sync(mask, value, offset, 8);
        if (lane >= offset && lower < value)
        {
            value = lower;
        }
    }
    return value;
}

__device__ inline ColdWallThermalReal coldWall1DPcrSolve
(
    ColdWallThermalReal lower,
    ColdWallThermalReal diagonal,
    ColdWallThermalReal upper,
    ColdWallThermalReal rightHandSide,
    const int lane,
    const unsigned int mask
)
{
    for (int stride = 1; stride < 8; stride *= 2)
    {
        const ColdWallThermalReal lowerLower =
            __shfl_up_sync(mask, lower, stride, 8);
        const ColdWallThermalReal lowerDiagonal =
            __shfl_up_sync(mask, diagonal, stride, 8);
        const ColdWallThermalReal lowerUpper =
            __shfl_up_sync(mask, upper, stride, 8);
        const ColdWallThermalReal lowerRightHandSide =
            __shfl_up_sync(mask, rightHandSide, stride, 8);
        const ColdWallThermalReal upperLower =
            __shfl_down_sync(mask, lower, stride, 8);
        const ColdWallThermalReal upperDiagonal =
            __shfl_down_sync(mask, diagonal, stride, 8);
        const ColdWallThermalReal upperUpper =
            __shfl_down_sync(mask, upper, stride, 8);
        const ColdWallThermalReal upperRightHandSide =
            __shfl_down_sync(mask, rightHandSide, stride, 8);
        const ColdWallThermalReal alpha = lane >= stride
          ? -lower/(lowerDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const ColdWallThermalReal beta = lane + stride < 8
          ? -upper/(upperDiagonal + GPU_REAL_MIN)
          : GPU_R(0.0);
        const ColdWallThermalReal nextLower = lane >= stride
          ? alpha*lowerLower
          : GPU_R(0.0);
        const ColdWallThermalReal nextUpper = lane + stride < 8
          ? beta*upperUpper
          : GPU_R(0.0);
        const ColdWallThermalReal nextDiagonal = diagonal
          + alpha*lowerUpper + beta*upperLower;
        const ColdWallThermalReal nextRightHandSide = rightHandSide
          + alpha*lowerRightHandSide + beta*upperRightHandSide;
        lower = nextLower;
        diagonal = nextDiagonal;
        upper = nextUpper;
        rightHandSide = nextRightHandSide;
    }
    return rightHandSide/(diagonal + GPU_REAL_MIN);
}

__device__ inline ColdWallThermalReal coldWall1DSolidFractionFromKnownTemperature
(ColdWallThermalReal temperatureK,const Foam::gpuThermal::ColdWallSolidificationParameters& parameters)
{ return coldWallCarryFraction(temperatureK,parameters); }
