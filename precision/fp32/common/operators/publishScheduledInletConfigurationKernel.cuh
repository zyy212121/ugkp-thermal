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

template<class Real>
__device__ Real sqr3(const Real x, const Real y, const Real z)
{
    return x*x + y*y + z*z;
}
