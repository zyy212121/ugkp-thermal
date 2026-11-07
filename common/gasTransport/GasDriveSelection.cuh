#pragma once
#include "GasDriveSelection.H"
#include <cuda_runtime.h>
namespace ugkwp { namespace drive_selection_detail {
enum Candidate : int {Skip,Bound,LayoutChanged,NonfiniteChange};
template<class History>
__global__ void initialize(History history,const GasDriveControls* control,GasDriveSelection* result){
    if(blockIdx.x||threadIdx.x)return;
    const auto c=*control;auto& out=*result;out.held=out.first=history.frames;out.error=GasDriveSelectionError::None;
    // A completed history ends at the requested window endpoint. This is the
    // original consumed-history contract, not a new step bound or retry gate.
    if(!history.frames||!(history.interval[history.frames-1].end==c.total)||c.begin<ugkp::CompensatedTime{}
        ||c.total<c.requested||!(c.begin<c.requested)||!::isfinite(c.tolerance)||c.tolerance<0
        ||!::isfinite(c.tractionScale)||!(c.tractionScale>0)){
        out.error=GasDriveSelectionError::InvalidControl;return;}
    for(std::size_t i=0;i<history.frames;++i)if(c.begin<history.interval[i].end){out.held=i;break;}
    if(out.held==history.frames)out.error=GasDriveSelectionError::NoSample;
}
template<int Ns,class History>
__global__ void compareCandidates(History history,const GasDriveControls* control,GasDriveSelection* result,int* candidates){
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
// The caller owns capacity and input/decision storage. All runtime selection,
// first-match precedence and interval decisions stay on device. Comparing
// candidate records is independent; the earliest qualifying record retains the
// native serial order, including unordered externally supplied histories.
template<int Ns,class History>
inline cudaError_t selectGasDriveInterval(History history,const GasDriveControls* controls,
    GasDriveSelection* result,int* candidates,cudaStream_t stream){
    drive_selection_detail::initialize<<<1,1,0,stream>>>(history,controls,result);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    if(history.frames){const auto needed=history.frames/128+(history.frames%128!=0);
        const unsigned blocks=static_cast<unsigned>(needed<65535?needed:65535);
        drive_selection_detail::compareCandidates<Ns><<<blocks,128,0,stream>>>(history,controls,result,candidates);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;}
    drive_selection_detail::finish<<<1,1,0,stream>>>(history,controls,candidates,result);
    return cudaGetLastError();
}
}
