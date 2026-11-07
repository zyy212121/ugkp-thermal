#pragma once
#include "gpu/ResidentMaterialSlice.H"
#include <cuda_runtime.h>
namespace chmt { namespace material_slice_detail {
__global__ void plan(ResidentMaterialDriveHistory h,const ugkp::IntervalSliceCursor* cursor,
    const ugkp::CompensatedTime* target,std::size_t externalFaces,ugkp::IntervalSlicePlan* result){
    if(blockIdx.x||threadIdx.x)return;
    auto p=ugkp::planIntervalSlice(h,*cursor,*target);
    // The host cursor checks this only when it has a history radiation vector.
    if(p.error==ugkp::IntervalSliceError::None&&p.hasRecords()&&h.faces&&externalFaces&&externalFaces!=h.faces)
        p.error=ugkp::IntervalSliceError::RadiationLayout;
    *result=p;
}
__global__ void prepare(ResidentMaterialSlice slice,MaterialSliceAccumulators candidate,
    const Real* external,std::size_t externalFaces,ugkp::CompensatedTime total,Real* radiation){
    const auto p=*slice.plan;if(p.error!=ugkp::IntervalSliceError::None)return;
    const auto h=slice.history;const auto count=h.faces>externalFaces?h.faces:externalFaces;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<count;f+=std::size_t(blockDim.x)*gridDim.x){
        if(f<h.faces){
            // Only the last partial record survives in the candidate cursor.
            for(unsigned k=0;k<3;++k){const auto local=3*f+k;ExchangePacket used{};
                if(!p.hasRecords())used=slice.used.packets[local];
                else if(p.end.record<p.past){
                    const auto r=p.end.record,slot=(r-p.begin.record)*h.faces*3+local;
                    if(r==p.begin.record)used=slice.used.packets[local];
                    if(slice.active(slot))addIntervalPacket(used,slice.packet(slot));
                }
                candidate.packets[local]=used;
            }
            Real used=slice.used.radiation[f],sum=0;
            for(std::size_t r=p.begin.record;r<p.past;++r){
                const bool last=ugkp::sliceRecordStop(h,p,r)==h.interval[r].end;
                const Real value=h.surface[r*h.faces+f].radiationEnergy;
                const Real part=last?value-used:ugkp::sliceRecordFraction(h,p,r)*value;
                used+=part;sum+=part;if(last)used=0;
            }
            candidate.radiation[f]=used;
            if(p.hasRecords())radiation[f]=sum;
        }
        if(f<externalFaces){
            Real used=slice.used.externalRadiation[f];
            const Real fraction=(p.target-p.begin.time).value()/total.value();
            const auto end=h.frames?h.interval[h.frames-1].end:ugkp::CompensatedTime{};
            const Real part=p.target==end?external[f]-used:external[f]*fraction;
            used+=part;candidate.externalRadiation[f]=used;
            if(!p.hasRecords()||!h.faces)radiation[f]=0;
            radiation[f]+=part;
        }
    }
}
__global__ void commit(const ugkp::IntervalSlicePlan* plan,MaterialSliceAccumulators candidate,
    MaterialSliceAccumulators current,std::size_t faces,std::size_t externalFaces,
    ugkp::IntervalSliceCursor* cursor,const int* accepted){
    if(plan->error!=ugkp::IntervalSliceError::None||(accepted&&!*accepted))return;
    if(!blockIdx.x&&!threadIdx.x)*cursor=plan->end;
    const auto count=faces>externalFaces?faces:externalFaces;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<count;f+=std::size_t(blockDim.x)*gridDim.x){
        if(f<faces){for(unsigned k=0;k<3;++k)current.packets[3*f+k]=candidate.packets[3*f+k];current.radiation[f]=candidate.radiation[f];}
        if(f<externalFaces)current.externalRadiation[f]=candidate.externalRadiation[f];
    }
}
inline unsigned blocks(std::size_t n){const auto count=n/128+(n%128!=0);return unsigned(count>65535?65535:(count?count:1));}
}
// Allocations and stream ownership belong to the material owner. Preparation
// and commit are separate so a rejected physical source never consumes history.
// No publication, host decision or temporary packet-history copy is required.
inline cudaError_t prepareResidentMaterialSlice(ResidentMaterialSlice slice,
    const ugkp::IntervalSliceCursor* cursor,const ugkp::CompensatedTime* target,
    MaterialSliceAccumulators candidate,const Real* external,std::size_t externalFaces,
    ugkp::CompensatedTime total,Real* radiation,ugkp::IntervalSlicePlan* plan,cudaStream_t stream){
    material_slice_detail::plan<<<1,1,0,stream>>>(slice.history,cursor,target,externalFaces,plan);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    slice.plan=plan;const auto count=slice.history.faces>externalFaces?slice.history.faces:externalFaces;
    if(count)material_slice_detail::prepare<<<material_slice_detail::blocks(count),128,0,stream>>>(slice,candidate,external,externalFaces,total,radiation);
    return cudaGetLastError();
}
inline cudaError_t commitResidentMaterialSlice(const ugkp::IntervalSlicePlan* plan,
    MaterialSliceAccumulators candidate,MaterialSliceAccumulators current,
    std::size_t faces,std::size_t externalFaces,ugkp::IntervalSliceCursor* cursor,
    const int* accepted,cudaStream_t stream){
    const auto count=faces>externalFaces?faces:externalFaces;
    material_slice_detail::commit<<<material_slice_detail::blocks(count),128,0,stream>>>(plan,candidate,current,faces,externalFaces,cursor,accepted);
    return cudaGetLastError();
}
}
