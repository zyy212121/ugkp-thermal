#pragma once
#ifndef GPU_GEOMETRY_ROUNDING_AWARE
#define GPU_GEOMETRY_ROUNDING_AWARE 0
#endif
__device__ GPU_OPERATOR_REAL facePlaneDistance(const DeviceState& s, int p, GPU_OPERATOR_REAL x, GPU_OPERATOR_REAL y, GPU_OPERATOR_REAL z)
{
#if GPU_GEOMETRY_ROUNDING_AWARE
    const int f = s.cellFaceId[p];
    return s.planeNx[p]*(x-s.faceCx[f]) + s.planeNy[p]*(y-s.faceCy[f]) + s.planeNz[p]*(z-s.faceCz[f]);
#else
    return s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
#endif
}
__device__ __forceinline__ GPU_OPERATOR_REAL cellClassificationTolerance(const DeviceState& s, int c)
{
    return GPU_OPERATOR_R(1e-9)*clampMin(s.cellLength[c], GPU_OPERATOR_R(1e-12));
}
__device__ GPU_OPERATOR_REAL faceClassificationTolerance(const DeviceState& s, int c, int p, GPU_OPERATOR_REAL x, GPU_OPERATOR_REAL y, GPU_OPERATOR_REAL z)
{
    const GPU_OPERATOR_REAL legacy = cellClassificationTolerance(s, c);
#if GPU_GEOMETRY_ROUNDING_AWARE
    const int f = s.cellFaceId[p];
    const GPU_OPERATOR_REAL scale = fabs(s.planeNx[p])*(fabs(x)+fabs(s.faceCx[f]))
        + fabs(s.planeNy[p])*(fabs(y)+fabs(s.faceCy[f]))
        + fabs(s.planeNz[p])*(fabs(z)+fabs(s.faceCz[f]));
    return fmax(legacy, GPU_OPERATOR_R(4)*FLT_EPSILON*scale);
#else
    return legacy;
#endif
}
__device__ GPU_OPERATOR_REAL insideFaceCoordinate(GPU_OPERATOR_REAL hit, GPU_OPERATOR_REAL normal, GPU_OPERATOR_REAL eps)
{
    const GPU_OPERATOR_REAL shifted = hit - eps*normal;
#if GPU_GEOMETRY_ROUNDING_AWARE
    return normal == GPU_OPERATOR_R(0) ? shifted : nextafterf(shifted, normal > GPU_OPERATOR_R(0) ? -INFINITY : INFINITY);
#else
    return shifted;
#endif
}
