#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ bool pointInsideCell(const DeviceState& s, const int c, const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL y, const GPU_OPERATOR_REAL z)
{
    const int start = s.cellPlaneStart[c];
    const int count = s.cellPlaneCount[c];
    const GPU_OPERATOR_REAL tol = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL dist =
            s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
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
    GPU_OPERATOR_REAL maxDist = -1.0e300;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL dist =
            s.planeNx[p]*x + s.planeNy[p]*y + s.planeNz[p]*z - s.planeD[p];
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
    const GPU_OPERATOR_REAL tol = 1.0e-9*clampMin(s.cellLength[c], 1.0e-12);
    int plane = -1;
    GPU_OPERATOR_REAL bestT = 2.0;
    for (int i = 0; i < count; ++i)
    {
        const int p = start + i;
        const GPU_OPERATOR_REAL d0 =
            s.planeNx[p]*x0 + s.planeNy[p]*y0 + s.planeNz[p]*z0 - s.planeD[p];
        const GPU_OPERATOR_REAL d1 =
            s.planeNx[p]*x1 + s.planeNy[p]*y1 + s.planeNz[p]*z1 - s.planeD[p];
        if (d1 <= tol)
        {
            continue;
        }

        const GPU_OPERATOR_REAL denom = d1 - d0;
        GPU_OPERATOR_REAL t = 1.0;
        if (denom > 1.0e-300)
        {
            t = (tol - d0)/denom;
        }
        t = clampRange(t, 0.0, 1.0);
        if (t < bestT)
        {
            bestT = t;
            plane = p;
        }
    }

    hitT = bestT <= 1.0 ? bestT : 1.0;
    if (plane < 0)
    {
        plane = mostViolatedPlane(s, c, x1, y1, z1);
    }
    return plane;
}
