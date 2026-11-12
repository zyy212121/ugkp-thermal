#pragma once
#include "SteadySamplingMoments.cuh"
#include "../../../common/gasTransport/GasResidentInterval.H"
#include "../../../common/GpuVectorSum.cuh"
// Parcel boundary inventory audit, before/after injection and face walking.
// Optional buffers are null for all existing transient executions.
struct SteadyParcelInventory {
    DeviceState* state;
    __device__ int count()const {
        return clampRange(*state->particleCountDevice,0,state->particleCapacity);
    }
    __device__ void add(int i,GpuReal* q)const {
        const DeviceState& s=*state;
        if(s.pStatus[i]!=1)return;
        const GpuReal m=s.pm[i],ux=s.pux[i],uy=s.puy[i],uz=s.puz[i];
        q[0]+=m;q[1]+=m*ux;q[2]+=m*uy;q[3]+=m*uz;
        q[4]+=m*(GpuReal(0.5)*(ux*ux+uy*uy+uz*uz)+GpuReal(1.5)*s.pTheta[i]);
        q[5]+=m*particleHeatFactorDevice(s)*s.pT[i];
    }
};
struct SteadyGasBoundaryFlux {
    DeviceState* state;double dt;
    __device__ int count()const {return state->nFaces-state->nInternalFaces;}
    __device__ void add(int i,GpuReal* q)const {
        const DeviceState& s=*state;const int f=s.nInternalFaces+i;
        if(coupledFaceNeighbour(s,f)>=0)return;
        const GpuReal intervalDt=GpuReal(ugkwp::gasResidentStepDt(s,dt));
        q[0]-=intervalDt*s.gasPhiRho[f];q[1]-=intervalDt*s.gasPhiRhoUx[f];
        q[2]-=intervalDt*s.gasPhiRhoUy[f];q[3]-=intervalDt*s.gasPhiRhoUz[f];
        q[4]-=intervalDt*s.gasPhiRhoE[f];
        if(ugkwp::mixtureGasActive(s))
            for(int k=0;k<ugkwp::compiledGasSpecies;++k)
                q[5+k]-=intervalDt*s.gasSpecies.flux[k*s.nFaces+f];
    }
};
int configureSteadyAuditStorage(DeviceState* s)
{
    int parcelBlocks=0,gasBlocks=0;
    auto err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&parcelBlocks,
        ugkwp::vectorSumPartials<6,256,SteadyParcelInventory>,256,0);
    if(err==cudaSuccess)err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gasBlocks,
        ugkwp::vectorSumPartials<GpuSteadySamplingReport::gasFluxFields,256,SteadyGasBoundaryFlux>,256,0);
    if(err!=cudaSuccess){setLastError("steady audit reduction occupancy",err);return 1;}
    s->steadyAuditParts=s->multiprocessorCount*(parcelBlocks>gasBlocks?parcelBlocks:gasBlocks);
    // Separate slices allow independent gas/particle streams. Neither slice
    // depends on parcel capacity, so growth and graph replay cannot overrun it.
    return allocate(s->steadyAuditWorkspace,
        size_t(6+GpuSteadySamplingReport::gasFluxFields)*s->steadyAuditParts,
        "allocate resident steady audit partials");
}
int auditSteadyParcelInventory(DeviceState* s,int offset,GpuReal sign)
{
    if(!s->steadyBoundaryBudget||s->particleWorkGrid<=0)return 0;
    const int needed=1+(s->particleCapacity-1)/256;
    const int parts=needed<s->steadyAuditParts?needed:s->steadyAuditParts;
    const auto err=ugkwp::accumulateVectorSum<6>(SteadyParcelInventory{s->deviceState},
        parts,s->steadyAuditWorkspace,s->steadyBoundaryBudget+offset,sign,
        ugkp::gpu::particleRecordingStream(s));
    if(err!=cudaSuccess){setLastError("steady parcel boundary inventory audit",err);return 1;}
    return 0;
}
int auditSteadyGasBoundaryFlux(DeviceState* s,double dt)
{
    if(!s->steadyBoundaryBudget||s->nFaces==s->nInternalFaces)return 0;
    const int needed=1+(s->nFaces-s->nInternalFaces-1)/256;
    const int parts=needed<s->steadyAuditParts?needed:s->steadyAuditParts;
    const auto err=ugkwp::accumulateVectorSum<GpuSteadySamplingReport::gasFluxFields>(
        SteadyGasBoundaryFlux{s->deviceState,dt},parts,
        s->steadyAuditWorkspace+size_t(6)*s->steadyAuditParts,
        s->steadyBoundaryBudget,GpuReal(1),s->gasCaptureStream);
    if(err!=cudaSuccess){setLastError("steady gas RK boundary flux audit",err);return 1;}
    return 0;
}
