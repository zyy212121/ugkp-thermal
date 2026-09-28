#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void accumulateMobilePackingMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int nParticles =
        clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        if (s.pStatus[i] != 1)
        {
            continue;
        }
        const int c = s.pCellId[i];
        if (c < 0 || c >= s.nCells)
        {
            continue;
        }
        const GPU_OPERATOR_REAL m = clampMin(finiteOr(s.pm[i], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
        if (s.pStuck[i] != 0)
        {
            atomicAdd(&s.packingStuckRho[c], m);
            continue;
        }
        const GPU_OPERATOR_REAL ux = GPU_OPERATOR_R(0.5)*
        (
            finiteOr(s.puxOld[i], s.pux[i]) + finiteOr(s.pux[i], GPU_OPERATOR_R(0.0))
        );
        const GPU_OPERATOR_REAL uy = GPU_OPERATOR_R(0.5)*
        (
            finiteOr(s.puyOld[i], s.puy[i]) + finiteOr(s.puy[i], GPU_OPERATOR_R(0.0))
        );
        const GPU_OPERATOR_REAL uz = GPU_OPERATOR_R(0.5)*
        (
            finiteOr(s.puzOld[i], s.puz[i]) + finiteOr(s.puz[i], GPU_OPERATOR_R(0.0))
        );
        atomicAdd(&s.mobilePackingRho[c], m);
        atomicAdd(&s.mobilePackingMomX[c], m*ux);
        atomicAdd(&s.mobilePackingMomY[c], m*uy);
        atomicAdd(&s.mobilePackingMomZ[c], m*uz);
    }
}
