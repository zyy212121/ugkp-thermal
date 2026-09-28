#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void clearParticleMomentsAndCountsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c > s.nCells)
    {
        return;
    }
    s.cellParticleCount[c] = 0;
    if (c == s.nCells)
    {
        return;
    }
    s.momRhoP[c] = GPU_OPERATOR_R(0.0);
    s.momRhoUPx[c] = GPU_OPERATOR_R(0.0);
    s.momRhoUPy[c] = GPU_OPERATOR_R(0.0);
    s.momRhoUPz[c] = GPU_OPERATOR_R(0.0);
    s.momRhoEP[c] = GPU_OPERATOR_R(0.0);
    s.momRhoPD[c] = GPU_OPERATOR_R(0.0);
    s.momRhoHpP[c] = GPU_OPERATOR_R(0.0);
}
