#pragma once
#include "ParticleRadiationResponse.H"
#include "AnchoredParticleRadiationResponse.H"
#include <cuda_runtime.h>
namespace ugkp { namespace gpu {
namespace anchoredSnapshotDetail {

template<class State,class View>
__device__ auto copyCarry(State& state,const View& view,int i,int)
 ->decltype(state.specificEnthalpyResidual=view.enthalpyCarry[i],void()) {
    state.specificEnthalpyResidual=view.enthalpyCarry?view.enthalpyCarry[i]:0;
}
template<class State,class View>
__device__ void copyCarry(State&,const View&,int,long) {}
}





__global__ void captureAnchoredRadiationEntriesKernel
(ParticleRadiationResponseView borrowed,const int* cellOffsets,const int* indices,
 int cells,RadiationEntryParticle<GpuReal>* entries,int* sourceParticleIndex){
    const auto view=resolveParticleRadiationResponse(borrowed);
    const int slots=cellOffsets[cells];
    for(int slot=blockIdx.x*blockDim.x+threadIdx.x;slot<slots;slot+=blockDim.x*gridDim.x){
        const int i=indices?indices[slot]:slot;
        RadiationEntryParticle<GpuReal> entry{};
        entry.mass=view.mass[i];entry.thermal.temperature=view.temperature[i];
        anchoredSnapshotDetail::copyCarry(entry.thermal,view,i,0);
        entry.eligible=view.status[i]==1&&view.wallState[i]==0;
        entries[slot]=entry;sourceParticleIndex[slot]=i;
    }
}


}}
