#pragma once
#include "GpuOperatorContract.cuh"
__device__ GPU_OPERATOR_GEOMETRY facePlaneDistance(const DeviceState& s, int p, GPU_OPERATOR_GEOMETRY x, GPU_OPERATOR_GEOMETRY y, GPU_OPERATOR_GEOMETRY z)
{
    return s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
}
__device__ __forceinline__ GPU_OPERATOR_GEOMETRY cellClassificationTolerance(const DeviceState& s, int c)
{
    return GPU_OPERATOR_G(1e-9)*clampMin(s.cellLength[c], GPU_OPERATOR_G(1e-12));
}
__device__ GPU_OPERATOR_GEOMETRY faceClassificationTolerance(const DeviceState& s, int c, int p, GPU_OPERATOR_GEOMETRY x, GPU_OPERATOR_GEOMETRY y, GPU_OPERATOR_GEOMETRY z)
{
    const GPU_OPERATOR_GEOMETRY legacy = cellClassificationTolerance(s, c);
    return legacy;
}
__device__ GPU_OPERATOR_GEOMETRY insideFaceCoordinate(GPU_OPERATOR_GEOMETRY hit, GPU_OPERATOR_GEOMETRY normal, GPU_OPERATOR_GEOMETRY eps)
{
    const GPU_OPERATOR_GEOMETRY shifted = hit - eps*normal;
    return shifted;
}
