#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL linearScheduledValueDevice
(
    const GPU_OPERATOR_TIME* times,
    const GPU_OPERATOR_REAL* values,
    const int count,
    const GPU_OPERATOR_TIME simulationTime
)
{
    if (count <= 0 || times == nullptr || values == nullptr)
    {
        return GPU_OPERATOR_R(0.0);
    }
    if (simulationTime <= times[0])
    {
        return values[0];
    }
    if (simulationTime >= times[count - 1])
    {
        return values[count - 1];
    }
    int low = 0;
    int high = count - 1;
    while (low + 1 < high)
    {
        const int middle = low + (high - low)/2;
        if (times[middle] <= simulationTime)
        {
            low = middle;
        }
        else
        {
            high = middle;
        }
    }
    const GPU_OPERATOR_TIME t0 = times[low];
    const GPU_OPERATOR_TIME t1 = times[high];
    const GPU_OPERATOR_REAL fraction = clampRange
    (
        (simulationTime - t0)/clampMin(t1 - t0, OfSmall),
        GPU_OPERATOR_R(0.0),
        GPU_OPERATOR_R(1.0)
    );
    return values[low] + fraction*(values[high] - values[low]);
}

__device__ bool scheduledInletFaceDevice
(
    const DeviceState& s,
    const int face
)
{
    return
        s.nScheduledInletFaces > 0
     && s.scheduledInletFaceMask != nullptr
     && face >= s.nInternalFaces
     && face < s.nFaces
     && s.scheduledInletFaceMask[face] != 0;
}

__device__ GPU_OPERATOR_REAL scheduledPressureDevice
(
    const DeviceState& s,
    const GPU_OPERATOR_TIME simulationTime
)
{
    return linearScheduledValueDevice
    (
        s.pressureScheduleTimes,
        s.pressureScheduleValues,
        s.nPressureScheduleRows,
        simulationTime
    );
}
