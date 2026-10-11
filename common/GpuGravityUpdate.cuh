#pragma once
#include "gasTransport/GasEnergy.H"
template<bool SanitizeInputs>
__device__ __forceinline__ void applyGasGravityCell
(DeviceState& s, const int c, const GPU_OPERATOR_TIME dt)
{
    using Energy = ugkwp::GasEnergyScalar<DeviceState>;
    GPU_OPERATOR_REAL rho=s.rho[c], mx=s.rhoUx[c], my=s.rhoUy[c], mz=s.rhoUz[c];
    Energy energy=s.rhoE[c];
    if constexpr (SanitizeInputs)
    {
        rho=clampMin(finiteOr(rho,s.rhoMin),s.rhoMin);
        mx=finiteOr(mx,GPU_OPERATOR_R(0.0)); my=finiteOr(my,GPU_OPERATOR_R(0.0)); mz=finiteOr(mz,GPU_OPERATOR_R(0.0));
        energy=ugkwp::gasEnergyFiniteOr(energy,Energy(0));
    }
    const GPU_OPERATOR_REAL gx=s.gravityX, gy=s.gravityY, gz=s.gravityZ;
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
    {
        const GPU_OPERATOR_REAL g2=sqr3(gx,gy,gz);
        const GPU_OPERATOR_REAL work=dt*(mx*gx+my*gy+mz*gz);
        s.rhoE[c]=energy+work+GPU_OPERATOR_R(0.5)*rho*dt*dt*g2;
    }
    else
    {
    const Energy g2=Energy(gx)*Energy(gx)+Energy(gy)*Energy(gy)+Energy(gz)*Energy(gz);
    const Energy work=Energy(dt)*(Energy(mx)*Energy(gx)+Energy(my)*Energy(gy)+Energy(mz)*Energy(gz));
    s.rhoE[c]=energy+work+Energy(0.5)*Energy(rho)*Energy(dt)*Energy(dt)*g2;
    }
    s.rhoUx[c]=mx+rho*gx*dt;
    s.rhoUy[c]=my+rho*gy*dt;
    s.rhoUz[c]=mz+rho*gz*dt;
}
