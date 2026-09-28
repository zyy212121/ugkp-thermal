#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void clearParticleMomentsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
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

__global__ void initialiseEpsGPrevKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const GPU_OPERATOR_REAL eps = clampRange(finiteOr(s.epsS[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    s.epsGPrev[c] = GPU_OPERATOR_R(1.0) - eps;
}

__global__ void initialiseThetaDragAlphaKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    s.thetaDragAlpha[c] = GPU_OPERATOR_R(1.0);
}
