#pragma once
#include "ParticleRadiationResponse.H"
#include "AnchoredParticleRadiationResponse.H"
#include <cuda_runtime.h>
namespace ugkp { namespace gpu {
namespace anchoredSnapshotDetail {

template<class State,class View>
__host__ __device__ auto copyCarry(State& state,const View& view,int i,int)
 ->decltype(state.specificEnthalpyResidual=view.enthalpyCarry[i],void()) {
    state.specificEnthalpyResidual=view.enthalpyCarry?view.enthalpyCarry[i]:0;
}
template<class State,class View>
__host__ __device__ void copyCarry(State&,const View&,int,long) {}
}





__host__ __device__ inline RadiationEntryParticle<GpuReal> readRadiationEntry(
    const ParticleRadiationResponseView& view,int particle)
{
    RadiationEntryParticle<GpuReal> entry{};
    entry.mass=view.mass[particle];
    entry.thermal.temperature=view.temperature[particle];
    anchoredSnapshotDetail::copyCarry(entry.thermal,view,particle,0);
    entry.eligible=view.status[particle]==1&&view.wallState[particle]==0;
    return entry;
}

__global__ void captureAnchoredRadiationEntriesKernel
(ParticleRadiationResponseView borrowed,const int* cellOffsets,const int* indices,
 int cells,RadiationEntryParticle<GpuReal>* entries,int* sourceParticleIndex){
    const auto view=resolveParticleRadiationResponse(borrowed);
    const int slots=cellOffsets[cells];
    for(int slot=blockIdx.x*blockDim.x+threadIdx.x;slot<slots;slot+=blockDim.x*gridDim.x){
        const int i=view.cellOffsets||!indices?slot:indices[slot];
        entries[slot]=readRadiationEntry(view,i);
        if(sourceParticleIndex)sourceParticleIndex[slot]=i;
    }
}


}}
