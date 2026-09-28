#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void normalizeParticleMomentsAtomicKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const GPU_OPERATOR_REAL invV = GPU_OPERATOR_R(1.0)/clampMin(s.V[c], s.rhoMin);
    s.momRhoP[c] *= invV;
    s.momRhoUPx[c] *= invV;
    s.momRhoUPy[c] *= invV;
    s.momRhoUPz[c] *= invV;
    s.momRhoEP[c] *= invV;
    s.momRhoPD[c] *= invV;
    s.momRhoHpP[c] *= invV;
}
