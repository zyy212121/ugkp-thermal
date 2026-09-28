#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void applyGasGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }
    const GPU_OPERATOR_REAL rho = clampMin(finiteOr(s.rho[c], s.rhoMin), s.rhoMin);
    const GPU_OPERATOR_REAL oldMomX = finiteOr(s.rhoUx[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL oldMomY = finiteOr(s.rhoUy[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL oldMomZ = finiteOr(s.rhoUz[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL work =
        dt*(oldMomX*s.gravityX + oldMomY*s.gravityY + oldMomZ*s.gravityZ);
    const GPU_OPERATOR_REAL gravitySquared =
        sqr3(s.gravityX, s.gravityY, s.gravityZ);
    s.rhoUx[c] = oldMomX + rho*s.gravityX*dt;
    s.rhoUy[c] = oldMomY + rho*s.gravityY*dt;
    s.rhoUz[c] = oldMomZ + rho*s.gravityZ*dt;
    s.rhoE[c] = finiteOr(s.rhoE[c], GPU_OPERATOR_R(0.0))
              + work + GPU_OPERATOR_R(0.5)*rho*dt*dt*gravitySquared;
}
