#pragma once
template<bool SanitizeInputs>
__device__ __forceinline__ void applyGasGravityCell
(DeviceState& s, const int c, const GPU_OPERATOR_TIME dt)
{
    GPU_OPERATOR_REAL rho=s.rho[c], mx=s.rhoUx[c], my=s.rhoUy[c], mz=s.rhoUz[c], energy=s.rhoE[c];
    if constexpr (SanitizeInputs)
    {
        rho=clampMin(finiteOr(rho,s.rhoMin),s.rhoMin);
        mx=finiteOr(mx,GPU_OPERATOR_R(0.0)); my=finiteOr(my,GPU_OPERATOR_R(0.0)); mz=finiteOr(mz,GPU_OPERATOR_R(0.0));
        energy=finiteOr(energy,GPU_OPERATOR_R(0.0));
    }
    const GPU_OPERATOR_REAL gx=s.gravityX, gy=s.gravityY, gz=s.gravityZ;
    const GPU_OPERATOR_REAL g2=sqr3(gx,gy,gz);
    const GPU_OPERATOR_REAL work=dt*(mx*gx+my*gy+mz*gz);
    s.rhoE[c]=energy+work+GPU_OPERATOR_R(0.5)*rho*dt*dt*g2;
    s.rhoUx[c]=mx+rho*gx*dt;
    s.rhoUy[c]=my+rho*gy*dt;
    s.rhoUz[c]=mz+rho*gz*dt;
}
