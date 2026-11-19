#include "../gpu/WorkPlan.H"
#pragma once
#include "GasDriveSelection.H"
#include <cuda_runtime.h>
namespace ugkwp { namespace drive_selection_detail {
template<class History> __device__ auto resolveHistory(History history,int)->decltype(history.resolved()){return history.resolved();}
template<class History> __device__ History resolveHistory(History history,long){return history;}
enum Candidate : int {Skip,Bound,LayoutChanged,NonfiniteChange};
template<class History>
__global__ void initialize(History history,const GasDriveControls* control,GasDriveSelection* result){
    history=resolveHistory(history,0);
    if(blockIdx.x||threadIdx.x)return;
    const auto c=*control;auto& out=*result;out.held=out.first=history.frames;out.error=GasDriveSelectionError::None;


    if(!history.frames||!(history.interval[history.frames-1].end==c.total)||c.begin<ugkp::CompensatedTime{}
        ||c.total<c.requested||!(c.begin<c.requested)||!::isfinite(c.tolerance)||c.tolerance<0
        ||!::isfinite(c.tractionScale)||!(c.tractionScale>0)){
        out.error=GasDriveSelectionError::InvalidControl;return;}
    for(std::size_t i=0;i<history.frames;++i)if(c.begin<history.interval[i].end){out.held=i;break;}
    if(out.held==history.frames)out.error=GasDriveSelectionError::NoSample;
}
template<int Ns,class History>
__global__ void compareCandidates(History history,const GasDriveControls* control,GasDriveSelection* result,int* candidates){
    history=resolveHistory(history,0);
    if(result->error!=GasDriveSelectionError::None)return;
    const auto c=*control;const auto held=history.record(result->held);
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<history.frames;i+=std::size_t(blockDim.x)*gridDim.x){
        int decision=Skip;const auto begin=history.interval[i].begin;
        if(c.begin<begin&&begin<c.requested){const auto value=compareGasDrive<Ns>(held,history.record(i),c.tractionScale);
            if(!value.layoutMatches)decision=LayoutChanged;
            else if(!::isfinite(value.change))decision=NonfiniteChange;
            else if(value.change>c.tolerance||value.receiverChanged)decision=Bound;
        }
        candidates[i]=decision;
        if(decision!=Skip)atomicMin(&result->first,static_cast<unsigned long long>(i));
    }
}
template<class History>
__global__ void finish(History history,const GasDriveControls* control,const int* candidates,GasDriveSelection* result){
    history=resolveHistory(history,0);
    if(blockIdx.x||threadIdx.x||result->error!=GasDriveSelectionError::None)return;
    auto& out=*result;auto end=control->requested;
    if(out.first<history.frames){const int decision=candidates[out.first];
        if(decision==LayoutChanged){out.error=GasDriveSelectionError::LayoutChanged;return;}
        if(decision==NonfiniteChange){out.error=GasDriveSelectionError::NonfiniteChange;return;}
        end=history.interval[out.first].begin;
    }
    if(!(control->begin<end)){out.error=GasDriveSelectionError::NonAdvancing;return;}
    out.end=end;
}
}




template<int Ns,class History>
inline cudaError_t selectGasDriveInterval(History history,const GasDriveControls* controls,
    GasDriveSelection* result,int* candidates,cudaStream_t stream){
    drive_selection_detail::initialize<<<1,1,0,stream>>>(history,controls,result);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    if(history.frames){const auto needed=history.frames/ugkp::gpu::workPlan().cellBlockThreads+(history.frames%ugkp::gpu::workPlan().cellBlockThreads!=0);
        const unsigned blocks=static_cast<unsigned>(needed<65535?needed:65535);
        drive_selection_detail::compareCandidates<Ns><<<blocks,ugkp::gpu::workPlan().cellBlockThreads,0,stream>>>(history,controls,result,candidates);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;}
    drive_selection_detail::finish<<<1,1,0,stream>>>(history,controls,candidates,result);
    return cudaGetLastError();
}




namespace drive_selection_detail {
template<class History>
__global__ void initializeOrdered(History history,const GasDriveControls* control,
    const std::size_t* firstRecord,GasDriveSelection* result){
    history=resolveHistory(history,0);
    if(blockIdx.x||threadIdx.x)return;
    const auto c=*control;auto& out=*result;
    out.held=out.first=history.frames;out.error=GasDriveSelectionError::None;
    if(!history.frames||!(history.interval[history.frames-1].end==c.total)||c.begin<ugkp::CompensatedTime{}
        ||c.total<c.requested||!(c.begin<c.requested)||!::isfinite(c.tolerance)||c.tolerance<0
        ||!::isfinite(c.tractionScale)||!(c.tractionScale>0)||*firstRecord>history.frames){
        out.error=GasDriveSelectionError::InvalidControl;return;
    }
    for(std::size_t i=*firstRecord;i<history.frames;++i)
        if(c.begin<history.interval[i].end){out.held=i;break;}
    if(out.held==history.frames)out.error=GasDriveSelectionError::NoSample;
}
template<int Ns,class History>
__global__ void compareOrdered(History history,const GasDriveControls* control,
    GasDriveSelection* result,int* candidates){
    history=resolveHistory(history,0);
    if(result->error!=GasDriveSelectionError::None)return;
    const auto c=*control;const auto held=history.record(result->held);
    __shared__ unsigned long long first;
    for(std::size_t start=result->held;start<history.frames;start+=blockDim.x){
        if(!(history.interval[start].begin<c.requested))break;
        if(!threadIdx.x)first=history.frames;
        __syncthreads();
        const auto i=start+threadIdx.x;
        if(i<history.frames){
            int decision=Skip;const auto begin=history.interval[i].begin;
            if(c.begin<begin&&begin<c.requested){
                const auto value=compareGasDrive<Ns>(held,history.record(i),c.tractionScale);
                if(!value.layoutMatches)decision=LayoutChanged;
                else if(!::isfinite(value.change))decision=NonfiniteChange;
                else if(value.change>c.tolerance||value.receiverChanged)decision=Bound;
            }
            candidates[i]=decision;
            if(decision!=Skip)atomicMin(&first,static_cast<unsigned long long>(i));
        }
        __syncthreads();
        if(first<history.frames){if(!threadIdx.x)result->first=first;break;}
        __syncthreads();
    }
}
}
template<int Ns,class History>
inline cudaError_t selectOrderedGasDriveInterval(History history,const GasDriveControls* controls,
    const std::size_t* firstRecord,GasDriveSelection* result,int* candidates,cudaStream_t stream){
    drive_selection_detail::initializeOrdered<<<1,1,0,stream>>>(history,controls,firstRecord,result);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    drive_selection_detail::compareOrdered<Ns><<<1,ugkp::gpu::workPlan().cellBlockThreads,0,stream>>>(history,controls,result,candidates);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    drive_selection_detail::finish<<<1,1,0,stream>>>(history,controls,candidates,result);
    return cudaGetLastError();
}

}
