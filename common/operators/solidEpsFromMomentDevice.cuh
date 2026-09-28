#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL solidEpsFromMomentDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return 0.0;
    }

    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    return clampRange
    (
        rhoP/clampMin(s.rhoSolid, 1.0e-300),
        0.0,
        1.0
    );
}
