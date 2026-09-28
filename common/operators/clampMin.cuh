#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL clampMin(const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL lo)
{
    return x < lo ? lo : x;
}

__device__ bool finiteDevice(const GPU_OPERATOR_REAL x)
{
    return (x == x) && (fabs(x) < 1.0e300);
}

__device__ bool nonFiniteDevice(const GPU_OPERATOR_REAL x)
{
    return !finiteDevice(x);
}

__device__ GPU_OPERATOR_REAL finiteOr(const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL fallback)
{
    return finiteDevice(x) ? x : fallback;
}
