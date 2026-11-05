#pragma once
#include "SteadySamplingMoments.cuh"
// Parcel boundary inventory audit, before/after injection and face walking.
// Optional buffers are null for all existing transient executions.
__global__ void auditSteadyParcelInventoryKernel(DeviceState* sp,int offset,double sign)
{
    DeviceState& s=*sp;
    const int count=clampRange(*s.particleCountDevice,0,s.particleCapacity);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
    {
        if(s.pStatus[i]!=1)continue;
        const double m=s.pm[i],ux=s.pux[i],uy=s.puy[i],uz=s.puz[i];
        const double q[6]={m,m*ux,m*uy,m*uz,
            m*(0.5*(ux*ux+uy*uy+uz*uz)+1.5*s.pTheta[i]),
            m*particleHeatFactorDevice(s)*s.pT[i]};
        for(int k=0;k<6;++k)atomicAdd(s.steadyBoundaryBudget+offset+k,sign*q[k]);
    }
}

__global__ void auditSteadyGasBoundaryFluxKernel(DeviceState* sp,double dt)
{
    DeviceState& s=*sp;const int f=s.nInternalFaces+blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=s.nFaces||coupledFaceNeighbour(s,f)>=0)return;
    const double q[5]={s.gasPhiRho[f],s.gasPhiRhoUx[f],s.gasPhiRhoUy[f],s.gasPhiRhoUz[f],s.gasPhiRhoE[f]};
    for(int k=0;k<5;++k)atomicAdd(s.steadyBoundaryBudget+k,-dt*q[k]);
    if(ugkwp::mixtureGasActive(s))
        for(int k=0;k<ugkwp::compiledGasSpecies;++k)
            atomicAdd(s.steadyBoundaryBudget+5+k,-dt*s.gasSpecies.flux[k*s.nFaces+f]);
}

int auditSteadyParcelInventory(DeviceState* s,int offset,double sign)
{
    if(!s->steadyBoundaryBudget||s->particleWorkGrid<=0)return 0;
    auditSteadyParcelInventoryKernel<<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState,offset,sign);
    const auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("steady parcel boundary inventory audit",err);return 1;}
    return 0;
}

int auditSteadyGasBoundaryFlux(DeviceState* s,double dt)
{
    if(!s->steadyBoundaryBudget)return 0;
    const int block=s->fixedFaceBlockThreads;
    const int grid=(s->nFaces-s->nInternalFaces+block-1)/block;
    if(grid>0)auditSteadyGasBoundaryFluxKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,dt);
    const auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("steady gas RK boundary flux audit",err);return 1;}
    return 0;
}
