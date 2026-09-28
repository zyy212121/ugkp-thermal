#pragma once
// Sampling target depends on the completed pool, independent of its dispatch policy.
__device__ __forceinline__ void preparePoissonPoolSamplingCell(DeviceState& s, const int c)
{
    const int representatives = s.poolThermalCount[c];
    const GPU_OPERATOR_REAL poolMass =
        clampMin(finiteOr(s.poissonPoolMass[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));

    int target = 0;
    if (poolMass > GPU_OPERATOR_R(0.0) && representatives > 0)
    {
        target = representatives;
    }

    s.poissonPoolSampleTargetCount[c] = target;
}
