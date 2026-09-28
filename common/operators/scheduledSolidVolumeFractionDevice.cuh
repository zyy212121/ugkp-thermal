#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL scheduledSolidVolumeFractionDevice
(
    const DeviceState& s,
    const GPU_OPERATOR_REAL simulationTime
)
{
    return clampRange
    (
        linearScheduledValueDevice
        (
            s.volumeFractionScheduleTimes,
            s.volumeFractionScheduleValues,
            s.nVolumeFractionScheduleRows,
            simulationTime
        ),
        0.0,
        1.0 - OfSmall
    );
}
