#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void clearMobilePackingActivityCountsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *s.mobilePackingActiveCellCount = 0;
        *s.mobilePackingCorrectionCellCount = 0;
        *s.mobilePackingFrontierCurrentCount = 0;
        *s.mobilePackingFrontierNextCount = 0;
    }
}

__device__ int checkedMobilePackingCount
(
    const DeviceState& s,
    const int* count
)
{
    const int value = *count;
    if (value < 0 || value > s.nCells)
    {
        asm("trap;");
    }
    return value;
}

__global__ void clearMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    s.mobilePackingRho[c] = GPU_OPERATOR_R(0.0);
    s.packingStuckRho[c] = GPU_OPERATOR_R(0.0);
    s.mobilePackingMomX[c] = GPU_OPERATOR_R(0.0);
    s.mobilePackingMomY[c] = GPU_OPERATOR_R(0.0);
    s.mobilePackingMomZ[c] = GPU_OPERATOR_R(0.0);
    s.mobilePackingActiveCellMask[c] = 0;
    s.mobilePackingCorrectionCellMask[c] = 0;
    s.collisionalPressure[c] = GPU_OPERATOR_R(0.0);
    s.pressureKickScale[c] = GPU_OPERATOR_R(0.0);
    s.pressureDeltaMomX[c] = GPU_OPERATOR_R(0.0);
    s.pressureDeltaMomY[c] = GPU_OPERATOR_R(0.0);
    s.pressureDeltaMomZ[c] = GPU_OPERATOR_R(0.0);
}

__device__ void seedMobilePackingActivity(DeviceState& s, const int c)
{
    if (atomicCAS(&s.mobilePackingActiveCellMask[c], 0, 1) != 0)
    {
        return;
    }
    const int activeSlot = atomicAdd(s.mobilePackingActiveCellCount, 1);
    const int frontierSlot =
        atomicAdd(s.mobilePackingFrontierCurrentCount, 1);
    if (activeSlot < s.nCells)
    {
        s.mobilePackingActiveCellList[activeSlot] = c;
    }
    if (frontierSlot < s.nCells)
    {
        s.mobilePackingFrontierCurrent[frontierSlot] = c;
    }
}
