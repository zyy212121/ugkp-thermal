#pragma once
#include "GpuOperatorContract.cuh"
#include "GpuFaceGeometry.cuh"
#define GPU_GEOMETRY_LARGE GPU_OPERATOR_G(1.0e300)
__device__ bool pointInsideCell(const DeviceState& s, const int c, const GPU_OPERATOR_GEOMETRY x, const GPU_OPERATOR_GEOMETRY y, const GPU_OPERATOR_GEOMETRY z)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const GPU_OPERATOR_GEOMETRY tol = cellClassificationTolerance(s, c);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_GEOMETRY dist =
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
    const GPU_OPERATOR_GEOMETRY x,
    const GPU_OPERATOR_GEOMETRY y,
    const GPU_OPERATOR_GEOMETRY z
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    int plane = -1;
    GPU_OPERATOR_GEOMETRY maxDist = -GPU_GEOMETRY_LARGE;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_GEOMETRY dist =
            facePlaneDistance(s,p,x,y,z);
        if (dist > maxDist)
        {
            maxDist = dist;
            plane = p;
        }
    }
    return plane;
}

// The existing plane-intersection arithmetic is independent of bulk storage.
// In particular, its original binary64 denominator threshold must not pass
// through GPU_OPERATOR_TINY, whose scalar follows the particle/gas bulk type.
__device__ inline GPU_OPERATOR_GEOMETRY segmentPlaneIntersectionFraction
(
    const GPU_OPERATOR_GEOMETRY d0,
    const GPU_OPERATOR_GEOMETRY d1,
    const GPU_OPERATOR_GEOMETRY tol
)
{
    const GPU_OPERATOR_GEOMETRY denom = d1 - d0;
    GPU_OPERATOR_GEOMETRY t = GPU_OPERATOR_G(1.0);
    if (denom > GPU_OPERATOR_G(1.0e-300))
    {
        t = (tol - d0)/denom;
    }
    return clampRange(t, GPU_OPERATOR_G(0.0), GPU_OPERATOR_G(1.0));
}

__device__ int firstSegmentIntersection
(
    const DeviceState& s,
    const int c,
    const GPU_OPERATOR_GEOMETRY x0,
    const GPU_OPERATOR_GEOMETRY y0,
    const GPU_OPERATOR_GEOMETRY z0,
    const GPU_OPERATOR_GEOMETRY x1,
    const GPU_OPERATOR_GEOMETRY y1,
    const GPU_OPERATOR_GEOMETRY z1,
    GPU_OPERATOR_GEOMETRY& hitT
)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const GPU_OPERATOR_GEOMETRY tol = cellClassificationTolerance(s, c);
    int plane = -1;
    GPU_OPERATOR_GEOMETRY bestT = GPU_OPERATOR_G(2.0);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_GEOMETRY d0 =
            facePlaneDistance(s,p,x0,y0,z0);
        const GPU_OPERATOR_GEOMETRY d1 =
            facePlaneDistance(s,p,x1,y1,z1);
        if (d1 <= tol)
        {
            continue;
        }

        const GPU_OPERATOR_GEOMETRY t = segmentPlaneIntersectionFraction(d0, d1, tol);
        if (t < bestT)
        {
            bestT = t;
            plane = p;
        }
    }

    hitT = bestT <= GPU_OPERATOR_G(1.0) ? bestT : GPU_OPERATOR_G(1.0);
    if (plane < 0)
    {
        plane = mostViolatedPlane(s, c, x1, y1, z1);
    }
    return plane;
}
#undef GPU_GEOMETRY_LARGE
