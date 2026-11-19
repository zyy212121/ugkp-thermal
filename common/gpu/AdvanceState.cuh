#ifndef UGKP_GPU_ADVANCE_STATE_CUH
#define UGKP_GPU_ADVANCE_STATE_CUH
#include "AdvanceState.H"
#include "PhysicalEventClock.H"
#include "AdvanceExtension.H"
#include "GraphContinuationTask.cuh"
#include "ExecutionGraph.cuh"
#include "PinnedTransferQueue.cuh"
#include "../Diagnostics.H"
#include <cmath>
#include <memory>
#include <cuda/atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "../gasNumerics/GasWindowStep.H"
namespace ugkp { namespace gpu {
struct AdvanceReceipt {
    AdvanceState state;
    diagnostics::DeviceContext diagnostics;
    const int* activeCells;
    int baseReady;
};

struct AdvanceGraphStorage {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    cudaEvent_t completed=nullptr;AdvanceState* state=nullptr;AdvanceRequest* request=nullptr;
    double* maximumWorkspace=nullptr;std::size_t maximumParts=0;
    SteadyAdvanceControls steadyControls{};double* residualScales=nullptr;bool steadyConfigured=false;
    AdvanceRequest* hostRequest=nullptr;
    AdvanceReceipt* receipt=nullptr;
    AdvanceReceipt* hostReceipt=nullptr;
    AdvanceProgressPoint* hostProgress=nullptr;
    AdvanceProgressPoint* deviceProgress=nullptr;
    double acceptedTime=0,nominalDt=0;
    const bool reportBatches=std::getenv("UGKP_ADVANCE_STATS")!=nullptr;
    std::uint64_t batches=0,acceptedSteps=0,singleStepBatches=0,longestBatch=0;

    bool mayReachThermal(const AdvanceRequest& next) const {
        if(!extension)return false;
        if(next.thermalOutputStep>acceptedStepsAtStart &&
           next.thermalOutputStep-acceptedStepsAtStart<=next.maximumSteps)return true;
        const double event=std::min(next.thermalOutputTime,next.nextHeat);
        if(!reachedEventTime(next.boundary,event))return false;
        const double upper=next.adjustTimeStep
            ?std::max(nominalDt,next.maximumSteps==1?std::min(next.maximumDt,nominalDt*next.growthLimit):next.maximumDt)
            :nominalDt;
        const double n=static_cast<double>(next.maximumSteps);
        const double bound=acceptedTime+n*upper;
        const double roundoff=4*std::numeric_limits<double>::epsilon()*n*(std::fabs(bound)+std::fabs(acceptedTime));
        return reachedEventTime(bound+roundoff,event);
    }
    std::uint64_t acceptedStepsAtStart=0;

    PinnedTransferQueue transfers;
    GraphContinuationTask task;
    std::shared_ptr<AdvanceExtension> extension;
    bool pending=false,pureGas=false,includesCourant=false;unsigned long long epoch=0;
    int tasksReady=0,preparedKind=-1,baseReady=0,splitActive=0,injectionActive=0;
    void initialise(std::size_t parts){
        using executionGraphDetail::checked;
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);
        checked(cudaEventCreateWithFlags(&completed,cudaEventDisableTiming|cudaEventBlockingSync));
        checked(cudaMalloc(reinterpret_cast<void**>(&state),sizeof(*state)));
        checked(cudaMalloc(reinterpret_cast<void**>(&request),sizeof(*request)));
        checked(cudaMallocHost(reinterpret_cast<void**>(&hostRequest),sizeof(*hostRequest)));
        checked(cudaMalloc(reinterpret_cast<void**>(&receipt),sizeof(*receipt)));
        checked(cudaMallocHost(reinterpret_cast<void**>(&hostReceipt),sizeof(*hostReceipt)));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&hostProgress),advanceProgressCapacity*sizeof(*hostProgress),cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&deviceProgress),hostProgress,0));
        maximumParts=parts;
        checked(cudaMalloc(reinterpret_cast<void**>(&maximumWorkspace),(parts+1)*sizeof(double)));
    }
    void clearGraph(){
        if(executable){executionGraphDetail::checked(cudaGraphExecDestroy(executable));executable=nullptr;}
        if(graph){executionGraphDetail::checked(cudaGraphDestroy(graph));graph=nullptr;}
    }
    ~AdvanceGraphStorage(){

        task.reset();if(stream)cudaStreamSynchronize(stream);transfers.drain();
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(state)cudaFree(state);if(request)cudaFree(request);if(maximumWorkspace)cudaFree(maximumWorkspace);
        if(receipt)cudaFree(receipt);
        if(hostReceipt)cudaFreeHost(hostReceipt);
        if(hostRequest)cudaFreeHost(hostRequest);
        if(hostProgress)cudaFreeHost(hostProgress);
        if(residualScales)cudaFree(residualScales);
        if(reportBatches && batches)
            std::fprintf(stderr,"ADVANCE_BATCHES count=%llu steps=%llu single=%llu max=%llu\n",
                (unsigned long long)batches,(unsigned long long)acceptedSteps,
                (unsigned long long)singleStepBatches,(unsigned long long)longestBatch);
        if(completed)cudaEventDestroy(completed);if(stream)cudaStreamDestroy(stream);
    }
};
template<class Real>
__global__ void advanceMaximumPartials(const Real* values,int count,int parts,double* output){
    extern __shared__ double shared[];double value=-INFINITY;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=parts*blockDim.x){
        const double next=values[i];value=isnan(value)?value:isnan(next)?next:fmax(value,next);
    }
    shared[threadIdx.x]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){
        if(threadIdx.x<active/2){const double next=shared[threadIdx.x+(active+1)/2];
            auto& current=shared[threadIdx.x];current=isnan(current)?current:isnan(next)?next:fmax(current,next);}
        __syncthreads();
    }
    if(!threadIdx.x)output[blockIdx.x]=shared[0];
}
template<class Real>
inline void advanceMaximum(const Real* values,int count,int block,AdvanceGraphStorage& owner){
    const int parts=int(owner.maximumParts);
    advanceMaximumPartials<<<parts,block,block*sizeof(double),owner.stream>>>(values,count,parts,
        parts==1?owner.maximumWorkspace+parts:owner.maximumWorkspace);
    executionGraphDetail::checked(cudaGetLastError());
    if(parts==1)return;
    advanceMaximumPartials<<<1,block,block*sizeof(double),owner.stream>>>(owner.maximumWorkspace,parts,1,owner.maximumWorkspace+parts);
    executionGraphDetail::checked(cudaGetLastError());
}
__device__ inline bool advanceMayContinue(AdvanceState& state,const AdvanceRequest& request,
    const diagnostics::DeviceContext* diagnostics){
    if(diagnostics::hasFatal(diagnostics))state.stop=AdvanceStop::fatal;
    else if(state.stop==AdvanceStop::running&&state.stage==AdvanceStage::selectStep){
        if(state.time>=request.boundary-request.stopOffset*state.clockDt)state.stop=AdvanceStop::boundary;
        else if(state.submissionSteps>=request.maximumSteps)state.stop=AdvanceStop::stepLimit;
        else if(request.maximumThermalEvents&&state.submissionThermalEvents>=request.maximumThermalEvents)state.stop=AdvanceStop::boundary;
    }
    return state.stop==AdvanceStop::running;
}
__global__ void beginAdvanceSubmission(AdvanceState* state,const AdvanceRequest* request,
    const diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle loop){
    if(threadIdx.x)return;
    if(!state->capacityResuming){
        state->submissionSteps=0;state->submissionThermalEvents=0;state->submissionRadiationEvents=0;
        state->nextHeat=fmin(request->nextHeat,request->thermalOutputTime);
        state->progressCount=0;
        const auto interval=request->progressInterval;
        state->nextProgressStep=interval ? state->steps+interval-state->steps%interval : 0;
        state->residualReady=false;state->windowReady=false;
    }
    state->capacityResuming=false;

    if(state->stop==AdvanceStop::boundary||state->stop==AdvanceStop::stepLimit)state->stop=AdvanceStop::running;
    cudaGraphSetConditional(loop,advanceMayContinue(*state,*request,diagnostics));
}
__global__ void resumeAdvanceCapacity(AdvanceState* state){
    if(!threadIdx.x&&state->stop==AdvanceStop::capacity){state->stop=AdvanceStop::running;state->capacityResuming=true;}
}
__global__ void finishAdvanceSubmissionStep(AdvanceState* state,const AdvanceRequest* request,
    const diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle loop){
    if(!threadIdx.x)cudaGraphSetConditional(loop,advanceMayContinue(*state,*request,diagnostics));
}
__device__ inline bool advanceCourantDue(const AdvanceState& state,const AdvanceRequest& request){
    return request.adjustTimeStep && state.stepsSinceCourant>=request.courantInterval;
}
template<class State>
__global__ void prepareAdvanceCourant(State* physics,AdvanceState* state,const AdvanceRequest* request,
    cudaGraphConditionalHandle refresh){
    if(threadIdx.x)return;
    if(state->stage!=AdvanceStage::selectStep || !advanceCourantDue(*state,*request)){cudaGraphSetConditional(refresh,0);return;}
    auto& interval=physics->gasResidentInterval;
    interval.dt=state->dt;interval.begin=state->boundaryFrozen?state->boundaryTime:state->time;
    interval.end=__dadd_rn(interval.begin,state->dt);interval.middle=__dadd_rn(interval.begin,__dmul_rn(0.5,state->dt));
    interval.active=true;interval.frozen=state->boundaryFrozen;
    if(physics->diagnostics)diagnostics::beginOperation(*physics->diagnostics,
        state->acceptedSequence+1,state->steps+1,state->time+state->dt);
    cudaGraphSetConditional(refresh,1);
}
template<class State>
__device__ inline void prepareAdvanceStep(State* physics,AdvanceState* state,const AdvanceRequest* request,const double* maximum){
    if(state->stage!=AdvanceStage::selectStep)return;
    state->previousDt=state->dt;
    state->previousClockDt=state->clockDt;
    if(advanceCourantDue(*state,*request)){
        state->lastCourant=*maximum;
        if(state->lastCourant>request->small){
            const double proposed=state->dt*request->targetCourant/fmax(state->lastCourant,request->small);
            state->dt=ugkwp::selectCourantTimeStep(state->dt,proposed,request->growthLimit,request->maximumDt,true);


            if(request->adjustWrite&&!request->exactBoundary){
                const double remaining=fmax(0.0,request->nextWrite-state->time);
                const double count=remaining/state->dt;
                if(count<2147483647.0)state->dt=remaining/static_cast<int>(fmax(count,1.0)+0.99);
            }
        }
        state->stepsSinceCourant=1;
    }else state->stepsSinceCourant=min(state->stepsSinceCourant+1,request->courantInterval);
    state->clockDt=state->clock==AdvanceClock::physical?state->dt:1.0;
    if(state->clock!=AdvanceClock::physical&&request->adjustWrite){
        const double remaining=fmax(0.0,request->nextWrite-state->time);
        if(remaining<2147483647.0)state->clockDt=remaining/static_cast<int>(fmax(remaining,1.0)+0.99);
    }
    state->intervalEnd=__dadd_rn(state->time,state->clockDt);
    if(state->clock==AdvanceClock::physical&&request->exactBoundary){
        double deadline=request->boundary;
        if(state->nextHeat>state->time)deadline=eventDeadline(state->time,state->nextHeat,deadline);
        const auto step=stepToEvent(state->time,state->dt,deadline);
        state->clockDt=step.dt;state->intervalEnd=step.end;
    }


    const double physicsDt=state->clock==AdvanceClock::physical?state->clockDt:state->dt;
    state->stage=AdvanceStage::gasParticle;
    if(!(state->dt>0)||!isfinite(state->dt)||!(state->intervalEnd>state->time)){
        diagnostics::recordFatal(physics->diagnostics,diagnostics::makeRecord(diagnostics::Code::NonFinite,
            35001,0,0,diagnostics::EntityKind::None,-1,-1,state->dt,state->time));
        state->stop=AdvanceStop::fatal;
    }
    auto& gas=physics->gasResidentInterval;
    gas.dt=physicsDt;gas.begin=state->boundaryFrozen?state->boundaryTime:state->time;
    gas.end=state->clock==AdvanceClock::physical&&!state->boundaryFrozen?state->intervalEnd:__dadd_rn(gas.begin,physicsDt);
    gas.middle=__dadd_rn(gas.begin,__dmul_rn(0.5,physicsDt));
    gas.active=true;gas.frozen=state->boundaryFrozen;
    auto& particles=physics->particleResidentInterval;
    particles.dt=physicsDt;particles.end=state->boundaryFrozen?state->boundaryTime:state->intervalEnd;
    particles.active=true;particles.positionSequenceEnabled=state->clock==AdvanceClock::sampling;
    particles.positionSequence=state->positionSequence+1;
    if(physics->diagnostics)diagnostics::beginOperation(*physics->diagnostics,
        state->acceptedSequence+1,state->steps+1,state->intervalEnd);
}
template<class State>
__device__ inline void enterPhysicalBranch(const State* physics,const AdvanceState& state,
    cudaGraphConditionalHandle initial,cudaGraphConditionalHandle retained,bool split){
    const bool run=state.stop==AdvanceStop::running && state.stage==AdvanceStage::gasParticle
        && !diagnostics::hasFatal(physics->diagnostics);
    cudaGraphSetConditional(initial,run && (!split || !physics->preBaseDirectoryReady));
    if(split)cudaGraphSetConditional(retained,run && physics->preBaseDirectoryReady);
}
template<class State>
__global__ void selectAdvanceStep(State* physics,AdvanceState* state,const AdvanceRequest* request,
    const double* maximum,cudaGraphConditionalHandle initial,cudaGraphConditionalHandle retained,bool split){
    prepareAdvanceStep(physics,state,request,maximum);
#ifndef UGKWP_STEADY_APPLICATION
    enterPhysicalBranch(physics,*state,initial,retained,split);
#endif
}
template<class State>
__global__ void enterSteadyPhysicalBranch(State* physics,const AdvanceState* state,
    cudaGraphConditionalHandle initial,cudaGraphConditionalHandle retained,bool split){
    enterPhysicalBranch(physics,*state,initial,retained,split);
}
template<class State>
__device__ inline void commitPhysicalStep(State* physics,AdvanceState* state,const AdvanceRequest* request,
    AdvanceProgressPoint* progress){
    if(!diagnostics::hasFatal(physics->diagnostics)&&state->stop==AdvanceStop::running){
        state->previousTime=state->time;state->time=state->intervalEnd;
        ++state->steps;++state->submissionSteps;++state->acceptedSequence;
        if(state->steps==state->nextProgressStep){
            progress[state->progressCount++]={state->steps,state->time,state->clockDt};
            state->nextProgressStep+=request->progressInterval;
        }
        if(state->clock==AdvanceClock::sampling){state->samplingTime=__dadd_rn(state->samplingTime,state->dt);++state->positionSequence;}
        state->stage=AdvanceStage::selectStep;
    }
    physics->gasResidentInterval.active=false;physics->particleResidentInterval.active=false;

}
template<class State>
__global__ void finishPhysicalSegmentStep(State* physics,AdvanceState* state,const AdvanceRequest* request,
    const double* maximum,cudaGraphConditionalHandle segment,bool split,int base,bool thermal,AdvanceProgressPoint* progress){
    if(thermal && (reachedEventTime(state->intervalEnd,state->nextHeat,state->clockDt) || state->steps+1==request->thermalOutputStep)){
        cudaGraphSetConditional(segment,0);
        return;
    }
    commitPhysicalStep(physics,state,request,progress);
    const bool again=advanceMayContinue(*state,*request,physics->diagnostics)
        && !advanceCourantDue(*state,*request) && (!split || bool(physics->preBaseDirectoryReady)==bool(base));
    if(again)prepareAdvanceStep(physics,state,request,maximum);
    cudaGraphSetConditional(segment,again && state->stop==AdvanceStop::running);
}
template<class State>
__global__ void completeThermalSegment(State* physics,AdvanceState* state,const AdvanceRequest* request,
    cudaGraphConditionalHandle loop,AdvanceProgressPoint* progress){
    if(state->stage==AdvanceStage::accept)commitPhysicalStep(physics,state,request,progress);
    cudaGraphSetConditional(loop,advanceMayContinue(*state,*request,physics->diagnostics));
}
template<class T>
__device__ void copyReceiptField(T& destination,const T& source)
{
    static_assert(std::is_trivially_copyable<T>::value,"advance receipt fields use transfer-safe storage");
    auto* to=reinterpret_cast<unsigned char*>(&destination);
    const auto* from=reinterpret_cast<const unsigned char*>(&source);
    for(unsigned i=threadIdx.x;i<sizeof(T);i+=blockDim.x)to[i]=from[i];
}
template<class State>
__global__ void publishAdvanceReceipt(const State* physics,const AdvanceState* state,AdvanceReceipt* receipt){
    copyReceiptField(receipt->state,*state);
    copyReceiptField(receipt->diagnostics,*physics->diagnostics);
    if(threadIdx.x==0){
        receipt->activeCells=physics->pCellId;
        receipt->baseReady=physics->preBaseDirectoryReady;
    }
}
}}
#endif
