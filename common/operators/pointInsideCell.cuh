#pragma once
#include "GpuFaceGeometry.cuh"
#define GPU_GEOMETRY_LARGE GPU_OPERATOR_R(1.0e300)
__device__ bool pointInsideCell(const DeviceState& s, const int c, const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL y, const GPU_OPERATOR_REAL z)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const GPU_OPERATOR_REAL tol = cellClassificationTolerance(s, c);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL dist =
            facePlaneDistance(s,p,x,y,z);
        if (dist > tol)
        {
            return false;
        }
    }
    return true;
}

__device__ int mostViolatedPlane
(
    const DeviceState& s,
    const int c,
    const GPU_OPERATOR_REAL x,
    const GPU_OPERATOR_REAL y,
    const GPU_OPERATOR_REAL z
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    int plane = -1;
    GPU_OPERATOR_REAL maxDist = -GPU_GEOMETRY_LARGE;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL dist =
            facePlaneDistance(s,p,x,y,z);
        if (dist > maxDist)
        {
            maxDist = dist;
            plane = p;
        }
    }
    return plane;
}

__device__ int firstSegmentIntersection
(
    const DeviceState& s,
    const int c,
    const GPU_OPERATOR_REAL x0,
    const GPU_OPERATOR_REAL y0,
    const GPU_OPERATOR_REAL z0,
    const GPU_OPERATOR_REAL x1,
    const GPU_OPERATOR_REAL y1,
    const GPU_OPERATOR_REAL z1,
    GPU_OPERATOR_REAL& hitT
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const GPU_OPERATOR_REAL tol = cellClassificationTolerance(s, c);
    int plane = -1;
    GPU_OPERATOR_REAL bestT = GPU_OPERATOR_R(2.0);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL d0 =
            facePlaneDistance(s,p,x0,y0,z0);
        const GPU_OPERATOR_REAL d1 =
            facePlaneDistance(s,p,x1,y1,z1);
        if (d1 <= tol)
        {
            continue;
        }

        const GPU_OPERATOR_REAL denom = d1 - d0;
        GPU_OPERATOR_REAL t = GPU_OPERATOR_R(1.0);
        if (denom > GPU_OPERATOR_TINY(1.0e-300))
        {
            t = (tol - d0)/denom;
        }
        t = clampRange(t, GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
        if (t < bestT)
        {
            bestT = t;
            plane = p;
        }
    }

    hitT = bestT <= GPU_OPERATOR_R(1.0) ? bestT : GPU_OPERATOR_R(1.0);
    if (plane < 0)
    {
        plane = mostViolatedPlane(s, c, x1, y1, z1);
    }
    return plane;
}
#undef GPU_GEOMETRY_LARGE
