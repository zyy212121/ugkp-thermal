#pragma once
#include "gpu/GasWindowControl.H"
#include "gpu/ResidentMaterialDrive.H"
#include "gpu/GasTimeStepControl.H"
#include "gasTransport/GasCapabilities.H"
#include "../../../common/Diagnostics.H"
#include <cuda_runtime.h>
#include <limits>
namespace chmt { namespace gas_window_detail {
__device__ inline void stop(GasWindowProgress* progress,ugkp::diagnostics::DeviceContext* diagnostics,
    unsigned site,double observed,double reference=0){
    progress->stop=GasWindowStop::Fatal;
    ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(
        ugkp::diagnostics::Code::InvalidPhysicalState,site,26,0,
        ugkp::diagnostics::EntityKind::None,-1,-1,observed,reference));
}
template<class State>
__global__ void begin(State* physics,GasWindowProgress* progress,const GasWindowControls* controls,
    cudaGraphConditionalHandle loop){
    if(threadIdx.x||blockIdx.x)return;
    if(ugkp::diagnostics::hasFatal(physics->diagnostics))progress->stop=GasWindowStop::Fatal;
    else if(progress->elapsed==controls->total)progress->stop=GasWindowStop::Complete;
    else if(progress->batchFrames>=progress->batchCapacity)progress->stop=GasWindowStop::HistoryCapacity;
    else progress->stop=GasWindowStop::Running;
    cudaGraphSetConditional(loop,progress->stop==GasWindowStop::Running);
}
__device__ inline void setClock(ugkwp::GasResidentInterval& interval,Real begin,Real dt){
    interval.dt=dt;interval.begin=begin;

    interval.end=Real(begin+dt);interval.middle=Real(begin+Real(Real(.5)*dt));
    interval.active=true;interval.frozen=false;
}
template<class State>
__global__ void reference(State* physics,GasWindowProgress* progress,const GasWindowControls* controls,
    const ResidentWallClock* clock,const ResidentWallKnotTime* knots,int count,
    cudaGraphConditionalHandle refresh){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*progress;const auto c=*controls;
    const Real time=ugkp::absoluteTime(clock->origin,p.elapsed);
    ugkp::diagnostics::beginOperation(*physics->diagnostics,p.microSteps+1,c.acceptedStep,time);
    auto boundary=c.total;
    for(int i=0;i<count;++i)if(p.elapsed<knots[i].local&&knots[i].local<boundary)boundary=knots[i].local;
    p.remaining=ugkp::boundedTimeDuration(boundary,p.elapsed);
    if(!finite(c.cap)||!finite(c.maximum)||!finite(p.remaining)||!finite(p.seed)||p.seed<0
        ||!finite(c.growth)||c.courantPeriod<=0){
        stop(progress,physics->diagnostics,38003,p.seed,p.remaining);cudaGraphSetConditional(refresh,0);return;
    }
    p.requested=requestedGasMicrostep(c.adaptive,c.fixedDt,p.seed,c.cap,c.maximum);
    p.reference=minValue(p.requested,p.remaining);
    if(!finite(p.reference)||!(p.reference>0)||p.elapsed+p.reference==p.elapsed){
        stop(progress,physics->diagnostics,38003,p.reference,time);cudaGraphSetConditional(refresh,0);return;
    }
    setClock(physics->gasResidentInterval,time,p.reference);
    p.refreshed=c.adaptive&&(!p.courantReady||p.gasSteps%std::uint64_t(c.courantPeriod)==0);
    cudaGraphSetConditional(refresh,p.refreshed&&!ugkp::diagnostics::hasFatal(physics->diagnostics));
}
__device__ inline GasWindowCourant merge(GasWindowCourant a,GasWindowCourant b){
    return {a.value!=a.value?a.value:b.value!=b.value?b.value:fmax(a.value,b.value),a.unavailable||b.unavailable};
}
template<class Real>
__global__ void courantParts(const Real* field,const int* status,int count,GasWindowCourant* parts){
    extern __shared__ GasWindowCourant shared[];
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    shared[threadIdx.x]=c<count?GasWindowCourant{double(field[c]),status&&status[c]==int(ugkwp::GasTransportCode::InvalidStorage)}
        :GasWindowCourant{-std::numeric_limits<double>::infinity(),0};
    __syncthreads();

    for(unsigned stride=blockDim.x/2;stride;stride/=2){
        if(threadIdx.x<stride)shared[threadIdx.x]=merge(shared[threadIdx.x],shared[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)parts[blockIdx.x]=shared[0];
}
__global__ void courantFinish(const GasWindowCourant* parts,int count,GasWindowCourant* result){
    if(threadIdx.x||blockIdx.x)return;
    GasWindowCourant value{-std::numeric_limits<double>::infinity(),0};
    for(int i=0;i<count;++i)value=merge(value,parts[i]);*result=value;
}
template<class State>
__global__ void select(State* physics,GasWindowProgress* progress,const GasWindowControls* controls,
    const GasWindowCourant* courant,ResidentWallClock* clock,const ResidentWallKnotTime* knots,int count,
    CoupledGasInterval* interval,ugkwp::ParticleResidentInterval* particles,cudaGraphConditionalHandle physical){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*progress;const auto c=*controls;
    if(p.stop!=GasWindowStop::Running||ugkp::diagnostics::hasFatal(physics->diagnostics)){
        cudaGraphSetConditional(physical,0);return;
    }
    if(p.refreshed){
        if(courant->unavailable||!::isfinite(courant->value)||courant->value<0){
            stop(progress,physics->diagnostics,38004,courant->value);cudaGraphSetConditional(physical,0);return;
        }
        p.rate=Real(courant->value/p.reference);p.courantReady=true;
    }
    const Real stable=p.rate>0?c.cfl/p.rate:p.requested;
    const auto selected=selectGasMicrostep(c.adaptive,p.requested,p.remaining,stable,c.growth,
        minValue(c.cap,c.maximum),p.refreshed);
    p.dt=selected.accepted;p.nextDt=selected.next;
    if(!finite(p.dt)||!(p.dt>0)||p.elapsed+p.dt==p.elapsed){
        stop(progress,physics->diagnostics,38003,p.dt,p.reference);cudaGraphSetConditional(physical,0);return;
    }
    *interval=prepareCoupledWallInterval(*clock,knots,count,p.dt,c.gasGeometry+(c.moving?p.microSteps+1:0));
    const Real time=ugkp::absoluteTime(clock->origin,p.elapsed);
    setClock(physics->gasResidentInterval,time,p.dt);
    *particles={double(p.dt),double(Real(ugkp::absoluteTime(clock->origin,interval->localEnd))),true};
    cudaGraphSetConditional(physical,1);
}
template<class State>
__global__ void accept(State* physics,GasWindowProgress* progress,const GasWindowControls* controls,
    const CoupledGasInterval* interval,ResidentMaterialDriveInterval* history,cudaGraphConditionalHandle loop){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*progress;
    if(p.stop==GasWindowStop::Running&&!ugkp::diagnostics::hasFatal(physics->diagnostics)){
        history[p.batchFrames]={interval->localBegin,interval->localEnd,interval->actualDt};
        ++p.batchFrames;++p.microSteps;++p.gasSteps;p.elapsed=interval->localEnd;p.seed=p.nextDt;
    }
    physics->gasResidentInterval.active=false;
    if(ugkp::diagnostics::hasFatal(physics->diagnostics)||p.stop==GasWindowStop::Fatal)p.stop=GasWindowStop::Fatal;
    else if(p.elapsed==controls->total)p.stop=GasWindowStop::Complete;
    else if(p.batchFrames>=p.batchCapacity)p.stop=GasWindowStop::HistoryCapacity;
    cudaGraphSetConditional(loop,p.stop==GasWindowStop::Running);
}
}}
