#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void normalizeMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], OfVSmall);
    s.mobilePackingRho[c] *= invV;
    s.packingStuckRho[c] *= invV;
    s.mobilePackingMomX[c] *= invV;
    s.mobilePackingMomY[c] *= invV;
    s.mobilePackingMomZ[c] *= invV;
}
