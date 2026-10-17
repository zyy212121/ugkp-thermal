#pragma once
#include "gasTransport/GasEnergy.H"
#include "Diagnostics.H"
template<bool ApplyDensityFloor, bool Diagnose = true>
__device__ __forceinline__ void applyGasGravityCell
(DeviceState& s, const int c, const GPU_OPERATOR_TIME dt)
{
    if constexpr (Diagnose)
    {
        if(ugkp::diagnostics::hasFatal(s.diagnostics)) return;
    }
    using Energy = ugkwp::GasEnergyScalar<DeviceState>;
    GPU_OPERATOR_REAL rho=s.rho[c], mx=s.rhoUx[c], my=s.rhoUy[c], mz=s.rhoUz[c];
    Energy energy=s.rhoE[c];
    if constexpr (Diagnose)
    {
        const Energy inputs[]={Energy(rho),Energy(mx),Energy(my),Energy(mz),energy,
            Energy(s.gravityX),Energy(s.gravityY),Energy(s.gravityZ),Energy(dt)};
        for(int i=0;i<9;++i) if(!ugkwp::gasEnergyFinite(inputs[i]))
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,30500,305,0,
                EntityKind::Cell,c,i,double(inputs[i]),0,0,
                i==0?Quantity::Density:(i==4?Quantity::InternalEnergy:Quantity::LinearSystem)));
            return;
        }
    }
    // Retain the finite legacy density floor; never substitute invalid gas data.
    if constexpr (ApplyDensityFloor)
        rho=clampMin(rho,s.rhoMin);
    Energy updatedEnergy;
    const GPU_OPERATOR_REAL gx=s.gravityX, gy=s.gravityY, gz=s.gravityZ;
    if constexpr (std::is_same<Energy,GPU_OPERATOR_REAL>::value)
    {
        const GPU_OPERATOR_REAL g2=sqr3(gx,gy,gz);
        const GPU_OPERATOR_REAL work=dt*(mx*gx+my*gy+mz*gz);
        updatedEnergy=energy+work+GPU_OPERATOR_R(0.5)*rho*dt*dt*g2;
    }
    else
    {
    const Energy g2=Energy(gx)*Energy(gx)+Energy(gy)*Energy(gy)+Energy(gz)*Energy(gz);
    const Energy work=Energy(dt)*(Energy(mx)*Energy(gx)+Energy(my)*Energy(gy)+Energy(mz)*Energy(gz));
    updatedEnergy=energy+work+Energy(0.5)*Energy(rho)*Energy(dt)*Energy(dt)*g2;
    }
    const GPU_OPERATOR_REAL updatedMx=mx+rho*gx*dt;
    const GPU_OPERATOR_REAL updatedMy=my+rho*gy*dt;
    const GPU_OPERATOR_REAL updatedMz=mz+rho*gz*dt;
    if constexpr (Diagnose)
    {
        const Energy results[]={updatedEnergy,Energy(updatedMx),Energy(updatedMy),Energy(updatedMz)};
        for(int i=0;i<4;++i) if(!ugkwp::gasEnergyFinite(results[i]))
        {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,30501,305,0,
                EntityKind::Cell,c,i,double(results[i]),0,0,
                i==0?Quantity::InternalEnergy:Quantity::LinearSystem));
            return;
        }
    }
    s.rhoE[c]=updatedEnergy;
    s.rhoUx[c]=updatedMx;
    s.rhoUy[c]=updatedMy;
    s.rhoUz[c]=updatedMz;
}
