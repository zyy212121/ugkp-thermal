#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL clampRange(const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL lo, const GPU_OPERATOR_REAL hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
