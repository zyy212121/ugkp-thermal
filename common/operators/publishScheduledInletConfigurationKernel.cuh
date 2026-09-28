#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void publishScheduledInletConfigurationKernel
(
    DeviceState* sp,
    const int nFaces,
    int* faceMask,
    const GPU_OPERATOR_REAL inletTemperature,
    const int nPressureRows,
    GPU_OPERATOR_TIME* pressureTimes,
    GPU_OPERATOR_REAL* pressureValues,
    const int nVolumeFractionRows,
    GPU_OPERATOR_TIME* volumeFractionTimes,
    GPU_OPERATOR_REAL* volumeFractionValues,
    const int particlesMayBePresent
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    DeviceState& s = *sp;
    s.nScheduledInletFaces = nFaces;
    s.scheduledInletFaceMask = faceMask;
    s.scheduledInletTemperature = inletTemperature;
    s.nPressureScheduleRows = nPressureRows;
    s.pressureScheduleTimes = pressureTimes;
    s.pressureScheduleValues = pressureValues;
    s.nVolumeFractionScheduleRows = nVolumeFractionRows;
    s.volumeFractionScheduleTimes = volumeFractionTimes;
    s.volumeFractionScheduleValues = volumeFractionValues;
    if (particlesMayBePresent != 0)
    {
        s.particlesMayBePresent = true;
    }
}

__device__ GPU_OPERATOR_REAL sqr3(const GPU_OPERATOR_REAL x, const GPU_OPERATOR_REAL y, const GPU_OPERATOR_REAL z)
{
    return x*x + y*y + z*z;
}
