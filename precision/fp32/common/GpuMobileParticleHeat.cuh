#pragma once
#include "GpuParticleHeatMomentView.cuh"
#include "GpuParticleGasHeatRelaxation.cuh"
#include "GpuMaterialEnthalpyMoment.cuh"
// Retain the published macro state of free particles, including its accepted
// pressure closure. Subtract only represented contact particles from mixed cells.
__global__ void prepareMobileParticleHeatMomentsKernel(DeviceState* sp)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
    const auto m=particleCellHeatMoments(s,c);auto* out=s.particleMobileHeatMoments+6*c;
    out[0]=m.mass;out[1]=m.px;out[2]=m.py;out[3]=m.pz;out[4]=m.diameter;out[5]=m.enthalpy;
}
__global__ void subtractContactParticleHeatMomentsKernel(DeviceState* sp)
{
    DeviceState& s=*sp;const int n=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        // Capture membership when the cell heat source is formed. A later
        // collision may detach this parcel without putting it in that source.
        s.particleContactHeatEpoch[i]=s.pStatus[i]==1
            && s.pStuck[i]!=Foam::gpuThermal::particleWallMobile;
        if(!s.particleContactHeatEpoch[i])continue;
        const int c=s.pCellId[i];if(c<0||c>=s.nCells)continue;
        const auto m=clampMin(finiteOr(s.pm[i],GPU_R(0)),GPU_R(0));
        const GpuReal invV=GPU_R(1)/clampMin(GpuReal(s.V[c]),s.rhoMin);
        const auto d=clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),GPU_R(1e-12f));
        const auto t=clampRange(finiteOr(s.pT[i],s.TpMin),s.TpMin,s.TpMax);
        auto* out=s.particleMobileHeatMoments+6*c;
        atomicAdd(out,-m*invV);
        atomicAdd(out+1,-m*finiteOr(s.pux[i],GPU_R(0))*invV);
        atomicAdd(out+2,-m*finiteOr(s.puy[i],GPU_R(0))*invV);
        atomicAdd(out+3,-m*finiteOr(s.puz[i],GPU_R(0))*invV);
        atomicAdd(out+4,-m*d*invV);
        atomicAdd(out+5,-materialEnthalpyMomentFromParticle(s,i,m,t,0)*invV);
    }
}
__device__ void relaxMobileParticleThermalState
(DeviceState& s,int i,int c,GpuReal reynolds,GpuReal diameter,GpuTime dt)
{
    if(!s.solveParticleTemperature || !s.particleGasHeatTransferModelId)return;
    const auto law=particleCaloricDevice(s,i);
    const float oldT=clampRange(particleEffectiveTemperatureDevice(s,i),float(s.TpMin),float(s.TpMax));
    const auto gasT=clampRange(finiteOr(s.couplingTgasOld[c],s.TgasMin),s.TgasMin,GPU_R(1e30f));
    const auto rate=particleGasHeatRelaxationRate(s,c,gasT,reynolds,diameter,s.rhoSolid*law.specificHeat(oldT));
    // Same exponential relaxation as gasUGKP, published to one float temperature.
    const float delta=clampRange(-expm1(-clampMin(rate,GPU_R(0))*GpuReal(dt))*(float(gasT)-oldT),
        float(s.TpMin)-oldT,float(s.TpMax)-oldT);
    ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i]};
    const auto publication=ugkwp::publishParticleThermalHeat(state,law.enthalpyIncrement(oldT,delta),s.TpMin,s.TpMax,law);
    if(publication.valid){s.pT[i]=state.temperature;}
}
