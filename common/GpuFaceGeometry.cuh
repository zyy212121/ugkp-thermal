#pragma once
__device__ GPU_OPERATOR_REAL facePlaneDistance(const DeviceState& s, int p, GPU_OPERATOR_REAL x, GPU_OPERATOR_REAL y, GPU_OPERATOR_REAL z)
{
    return s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
}
__device__ __forceinline__ GPU_OPERATOR_REAL cellClassificationTolerance(const DeviceState& s, int c)
{
    return GPU_OPERATOR_R(1e-9)*clampMin(s.cellLength[c], GPU_OPERATOR_R(1e-12));
}
__device__ GPU_OPERATOR_REAL faceClassificationTolerance(const DeviceState& s, int c, int p, GPU_OPERATOR_REAL x, GPU_OPERATOR_REAL y, GPU_OPERATOR_REAL z)
{
    const GPU_OPERATOR_REAL legacy = cellClassificationTolerance(s, c);
    return legacy;
}
__device__ GPU_OPERATOR_REAL insideFaceCoordinate(GPU_OPERATOR_REAL hit, GPU_OPERATOR_REAL normal, GPU_OPERATOR_REAL eps)
{
    const GPU_OPERATOR_REAL shifted = hit - eps*normal;
    return shifted;
}
