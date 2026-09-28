#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool IncludeTurbulence>
__global__ void computeGasInternalFaceFluxKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= s.nFaces)
    {
        return;
    }

    s.gasPhiRho[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUx[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUy[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoUz[f] = GPU_OPERATOR_R(0.0);
    s.gasPhiRhoE[f] = GPU_OPERATOR_R(0.0);

    (void)dt;
    computeRiemannGasFaceFluxDevice<IncludeTurbulence>
    (
        s,
        f,
        s.gasPhiRho[f],
        s.gasPhiRhoUx[f],
        s.gasPhiRhoUy[f],
        s.gasPhiRhoUz[f],
        s.gasPhiRhoE[f]
    );
}

__global__ void enforcePeriodicGasFluxAntisymmetryKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (!isPeriodicFace(s, f))
    {
        return;
    }
    const int pair = s.facePeriodicPair[f];
    if (f > pair)
    {
        return;
    }

    const GPU_OPERATOR_REAL rho = GPU_OPERATOR_R(0.5)*(s.gasPhiRho[f] - s.gasPhiRho[pair]);
    const GPU_OPERATOR_REAL rhoUx = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUx[f] - s.gasPhiRhoUx[pair]);
    const GPU_OPERATOR_REAL rhoUy = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUy[f] - s.gasPhiRhoUy[pair]);
    const GPU_OPERATOR_REAL rhoUz = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoUz[f] - s.gasPhiRhoUz[pair]);
    const GPU_OPERATOR_REAL rhoE = GPU_OPERATOR_R(0.5)*(s.gasPhiRhoE[f] - s.gasPhiRhoE[pair]);
    s.gasPhiRho[f] = rho;
    s.gasPhiRhoUx[f] = rhoUx;
    s.gasPhiRhoUy[f] = rhoUy;
    s.gasPhiRhoUz[f] = rhoUz;
    s.gasPhiRhoE[f] = rhoE;
    s.gasPhiRho[pair] = -rho;
    s.gasPhiRhoUx[pair] = -rhoUx;
    s.gasPhiRhoUy[pair] = -rhoUy;
    s.gasPhiRhoUz[pair] = -rhoUz;
    s.gasPhiRhoE[pair] = -rhoE;
}
