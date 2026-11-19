#include <algorithm>
#include <mutex>
#include "ThermalEventGraph.H"
#include "PhysicalEventClock.H"
#include "../Diagnostics.H"
#include "ExecutionGraph.cuh"
#include "PinnedTransferQueue.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <limits>
namespace ugkp { namespace gpu {
using namespace executionGraphDetail;
namespace {
enum EventCondition {AnyEvent,RadiationEvent,PlainEvent,UnpublishedParticles,SourceFinished,SolidFinished};
__global__ void selectEvent(const ThermalEventState* state,const diagnostics::DeviceContext* diagnostics,
    int stage,EventCondition condition,const int* childStage,int childFinished,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;
    bool run=state->stage==stage&&!diagnostics::hasFatal(diagnostics);
    if(condition==RadiationEvent)run=run&&state->radiationDue;
    if(condition==PlainEvent)run=run&&!state->radiationDue;
    if(condition==UnpublishedParticles)run=run&&!state->particlePublished&&state->radiationDue;
    if(condition==SourceFinished||condition==SolidFinished)run=run&&childStage&&*childStage==childFinished;
    if(condition==SolidFinished)run=run&&state->particlePublished;
    cudaGraphSetConditional(handle,run);
}


__device__ unsigned long long eventClock(){
    unsigned long long value;asm volatile("mov.u64 %0, %globaltimer;" : "=l"(value));return value;
}
__global__ void setEventStage(ThermalEventState* state,int stage,unsigned long long* stamps){
    if(!threadIdx.x){state->stage=stage;stamps[stage]=eventClock();}
}
__global__ void publishEventParticles(ThermalEventState* state){if(!threadIdx.x)state->particlePublished=true;}
__global__ void finishEvent(ThermalEventState* state,const diagnostics::DeviceContext* diagnostics,unsigned long long* stamps){
    if(!threadIdx.x&&!diagnostics::hasFatal(diagnostics)){state->committed=true;state->stage=4;stamps[4]=eventClock();}
}
__global__ void selectThermalAdvance(ThermalEventState* event,AdvanceState* advance,
    const AdvanceRequest* request,const diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle condition,unsigned long long* stamps){
    if(threadIdx.x)return;
    if(diagnostics::hasFatal(diagnostics)){advance->stop=AdvanceStop::fatal;cudaGraphSetConditional(condition,0);return;}
    if(advance->stage==AdvanceStage::gasParticle){
        const double elapsed=advance->intervalEnd-event->completedTime;
        if(elapsed>0){
            stamps[0]=eventClock();stamps[1]=stamps[0];
            event->time=advance->intervalEnd;event->heatInterval=elapsed;
            event->radiationInterval=event->time-event->completedRadiationTime;
            ++event->sequence;
            const bool output=advance->steps+1==request->thermalOutputStep ||
                reachedEventTime(event->time,request->thermalOutputTime,advance->clockDt) ||
                advance->submissionThermalEvents+1==request->thermalOutputEvent;
            event->radiationDue=event->radiationEnabled && radiationDueAt(event->completedRadiationTime,event->time,
                request->scheduleOrigin,event->period,event->radiationFrequency,event->sequence,output);
            event->stage=0;event->committed=false;event->particlePublished=!event->radiationDue;
            advance->stage=AdvanceStage::thermal;
        }else advance->stage=AdvanceStage::accept;
    }
    cudaGraphSetConditional(condition,advance->stop==AdvanceStop::running&&advance->stage==AdvanceStage::thermal);
}
__global__ void acceptThermalAdvance(ThermalEventState* event,AdvanceState* advance,const AdvanceRequest* request,
    const diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||advance->stage!=AdvanceStage::thermal)return;
    if(diagnostics::hasFatal(diagnostics)){advance->stop=AdvanceStop::fatal;return;}
    if(!event->committed){advance->stop=AdvanceStop::capacity;return;}
    event->previousTime=event->completedTime;event->completedTime=event->time;event->completedStep=advance->steps+1;
    if(event->radiationDue){event->completedRadiationTime=event->time;event->lastRadiationInterval=event->radiationInterval;++advance->submissionRadiationEvents;}
    advance->thermalSequence=event->sequence;++advance->submissionThermalEvents;
    advance->nextHeat=fmin(request->thermalOutputTime,event->period>0
        ?nextPeriodicTime(request->scheduleOrigin,event->period,event->completedTime):event->completedTime);
    advance->nextRadiation=event->radiationEnabled && event->period>0
        ?nextPeriodicTime(request->scheduleOrigin,event->period*event->radiationFrequency,event->completedRadiationTime):0;
    advance->stage=AdvanceStage::accept;
}
template<class Build> cudaGraphNode_t eventBranch(cudaGraph_t graph,cudaGraphNode_t dependency,
    cudaStream_t stream,ThermalEventState* state,const diagnostics::DeviceContext* diagnostics,
    int stage,EventCondition condition,const int* childStage,int childFinished,Build build){
    cudaGraphConditionalHandle handle;
    checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        selectEvent<<<1,1,0,stream>>>(state,diagnostics,stage,condition,childStage,childFinished,handle);
    });
    cudaGraph_t body=nullptr;const auto node=addIf(graph,handle,{selected},body);build(body);return node;
}
}
ExecutionGraphNode appendThermalOperation(ExecutionGraph graph,ExecutionGraphNode dependency,
    void* stream,const std::function<void()>& operation){
    return appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},
        static_cast<cudaStream_t>(stream),operation);
}
struct ThermalEventGraph::Implementation {
    struct Report {ThermalEventState state;unsigned long long stamps[5];};
    Report *deviceReport=nullptr,*hostReport=nullptr;
    std::mutex collectMutex;
    std::vector<void*> inputStreams;
    void cacheInputStreams(){
        inputStreams.clear();
        const auto add=[&](void* source){
            if(std::find(inputStreams.begin(),inputStreams.end(),source)==inputStreams.end())
                inputStreams.push_back(source);
        };
        for(auto source:bindings.preparedInputStreams)add(source);
        add(stream);
        if(bindings.radiation){add(bindings.radiation->producerStream());add(bindings.material->producerStream());}
        add(bindings.contact?bindings.contact->producerStream():bindings.solid->producerStream());
    }
    ThermalEventBindings bindings;
    ThermalEventState* state=nullptr;
    unsigned long long* stamps=nullptr;
    cudaStream_t stream=nullptr;cudaEvent_t ready=nullptr;
    PinnedTransferQueue transfers;
    std::shared_future<void> completion;
    explicit Implementation(const ThermalEventBindings& input):bindings(input){}
    ~Implementation(){
        if(completion.valid()){try{completion.get();}catch(...){}}
        if(stream)cudaStreamSynchronize(stream);transfers.drain();
        if(hostReport)cudaFreeHost(hostReport);
        if(deviceReport)cudaFree(deviceReport);
        if(ready)cudaEventDestroy(ready);if(stream)cudaStreamDestroy(stream);
    }
};
ThermalEventGraph::ThermalEventGraph(const ThermalEventBindings& bindings)
    :data_(new Implementation(bindings)){
    if(bool(bindings.solid)==bool(bindings.contact))
        throw std::invalid_argument("thermal event requires one solid iteration owner");
    if(bool(bindings.radiation)!=bool(bindings.material)||!bindings.diagnostics
        ||!bindings.prepareInterface||!bindings.publishParticles||!bindings.commit)
        throw std::invalid_argument("thermal event dependency bindings are incomplete");
    auto& d=*data_;
    checked(cudaStreamCreateWithFlags(&d.stream,cudaStreamNonBlocking));d.transfers.bind(d.stream);
    checked(cudaEventCreateWithFlags(&d.ready,cudaEventDisableTiming|cudaEventBlockingSync));
    checked(cudaMalloc(reinterpret_cast<void**>(&d.deviceReport),sizeof(*d.deviceReport)));
    checked(cudaMallocHost(reinterpret_cast<void**>(&d.hostReport),sizeof(*d.hostReport)));
    d.state=&d.deviceReport->state;d.stamps=d.deviceReport->stamps;
    checked(cudaMemsetAsync(d.stamps,0,sizeof(d.deviceReport->stamps),d.stream));
    d.cacheInputStreams();
    setEvent({});
}
ThermalEventGraph::~ThermalEventGraph()=default;
ThermalEventState* ThermalEventGraph::deviceState()const noexcept{return data_->state;}
void ThermalEventGraph::setEvent(const ThermalEventState& input){
    awaitCompletion();auto value=input;
    if(value.radiationDue&&!data_->bindings.radiation)
        throw std::invalid_argument("radiation event has no radiation owner");
    value.stage=0;value.committed=false;value.particlePublished=!value.radiationDue;
    PinnedTransferQueue::Operation operation(data_->transfers);
    data_->transfers.upload(data_->state,&value,sizeof(value));
}
void ThermalEventGraph::bindRadiation(DiscreteOrdinateRadiation& radiation,ParticleRadiationMaterial& material){
    awaitCompletion();data_->bindings.radiation=&radiation;data_->bindings.material=&material;data_->cacheInputStreams();
}
void ThermalEventGraph::orderPreparedInputs(void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    for(auto source:d.inputStreams){
        if(source==executionStream)continue;
        checked(cudaEventRecord(d.ready,static_cast<cudaStream_t>(source)));
        checked(cudaStreamWaitEvent(stream,d.ready,0));
    }
}
ExecutionGraphNode ThermalEventGraph::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,
    AdvanceState* advance,const AdvanceRequest* request,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    cudaGraphConditionalHandle condition;checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
    auto selected=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        selectThermalAdvance<<<1,1,0,stream>>>(d.state,advance,request,d.bindings.diagnostics,condition,d.stamps);
    });
    cudaGraph_t body=nullptr;
    const auto event=addIf(graph,condition,{selected},body);
    const auto committed=appendTo(body,nullptr);
    appendRecorded(body,{committed},stream,[&]{
        acceptThermalAdvance<<<1,1,0,stream>>>(d.state,advance,request,d.bindings.diagnostics);
    });
    return event;
}
ExecutionGraphNode ThermalEventGraph::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency){
    auto& d=*data_;auto& b=d.bindings;
    const auto branch=[&](cudaGraph_t target,cudaGraphNode_t before,int stage,EventCondition condition,
        const int* child,int finished,auto build){
        return eventBranch(target,before,d.stream,d.state,b.diagnostics,stage,condition,child,finished,build);
    };
    cudaGraphNode_t initialized=dependency;
    if(b.radiation){
        initialized=branch(graph,initialized,0,RadiationEvent,nullptr,0,[&](cudaGraph_t body){
            auto node=b.beforeRadiation?b.beforeRadiation(body,nullptr,d.stream):nullptr;
            node=b.material->appendWindowStartTo(body,node);
            node=b.radiation->appendInputsTo(body,node);
            appendRecorded(body,{node},d.stream,[&]{setEventStage<<<1,1,0,d.stream>>>(d.state,1,d.stamps);});
        });
    }
    initialized=branch(graph,initialized,0,PlainEvent,nullptr,0,[&](cudaGraph_t body){
        appendRecorded(body,{},d.stream,[&]{setEventStage<<<1,1,0,d.stream>>>(d.state,2,d.stamps);});
    });
    auto radiated=initialized;
    if(b.radiation){
        radiated=branch(graph,radiated,1,AnyEvent,nullptr,0,[&](cudaGraph_t body){
            b.radiation->appendSourceTo(body);
        });
        radiated=branch(graph,radiated,1,SourceFinished,b.radiation->continuationStage(),4,[&](cudaGraph_t body){
            auto node=b.radiation->appendDiagnosticsTo(body,nullptr);
            node=b.material->appendWindowFinishTo(body,node,b.radiation->deviceView());
            appendRecorded(body,{node},d.stream,[&]{setEventStage<<<1,1,0,d.stream>>>(d.state,2,d.stamps);});
        });
    }
    const auto prepared=branch(graph,radiated,2,AnyEvent,nullptr,0,[&](cudaGraph_t body){
        const auto node=b.prepareInterface(body,nullptr,d.stream);
        appendRecorded(body,{node},d.stream,[&]{
            if(b.contact)b.contact->resetContinuation(d.stream);else b.solid->resetContinuation(d.stream);
            setEventStage<<<1,1,0,d.stream>>>(d.state,3,d.stamps);
        });
    });


    cudaGraphNode_t particles=nullptr;
    if(b.radiation){
        particles=branch(graph,prepared,3,UnpublishedParticles,nullptr,0,[&](cudaGraph_t body){
            const auto node=b.publishParticles(body,nullptr,d.stream);
            appendRecorded(body,{node},d.stream,[&]{publishEventParticles<<<1,1,0,d.stream>>>(d.state);});
        });
    }
    const auto solid=branch(graph,prepared,3,AnyEvent,nullptr,0,[&](cudaGraph_t body){
        if(b.contact)b.contact->appendTo(body);else b.solid->appendTo(body);
    });
    cudaGraphNode_t joined=solid;
    if(particles){
        const cudaGraphNode_t children[]={particles,solid};
        checked(cudaGraphAddEmptyNode(&joined,graph,children,2));
    }
    return branch(graph,joined,3,SolidFinished,
        b.contact?b.contact->continuationStage():b.solid->continuationStage(),3,[&](cudaGraph_t body){
        const auto node=b.commit(body,nullptr,d.stream);
        appendRecorded(body,{node},d.stream,[&]{finishEvent<<<1,1,0,d.stream>>>(d.state,b.diagnostics,d.stamps);});
    });
}
bool ThermalEventGraph::servicePendingCapacity(){
    auto& b=data_->bindings;bool serviced=b.radiation&&b.radiation->servicePendingCapacity();
    const bool solid=b.contact?b.contact->servicePendingCapacity():b.solid->servicePendingCapacity();
    return serviced||solid;
}
void ThermalEventGraph::acceptExternalCompletion(const std::shared_future<void>& completion){
    auto& d=*data_;d.completion=completion;auto& b=d.bindings;
    if(b.radiation){b.radiation->acceptExternalCompletion(completion);b.material->acceptExternalCompletion(completion);}
    if(b.contact)b.contact->acceptExternalCompletion(completion);else b.solid->acceptExternalCompletion(completion);
}
void ThermalEventGraph::acceptExternalExecution(void* stream){
    auto& d=*data_;auto& b=d.bindings;
    checked(cudaEventRecord(d.ready,static_cast<cudaStream_t>(stream)));
    checked(cudaStreamWaitEvent(d.stream,d.ready,0));
    if(b.radiation){b.radiation->acceptExternalExecution(stream);b.material->acceptExternalExecution(stream);}
    if(b.contact)b.contact->acceptExternalExecution(stream);else b.solid->acceptExternalExecution(stream);
}
void ThermalEventGraph::awaitCompletion()const{
    auto& d=*data_;if(d.completion.valid())d.completion.get();
    checked(cudaStreamSynchronize(d.stream));
}
ThermalEventState ThermalEventGraph::collect()const{
    auto& d=*data_;
    std::lock_guard<std::mutex> lock(d.collectMutex);
    if(d.completion.valid())d.completion.get();
    checked(cudaMemcpyAsync(d.hostReport,d.deviceReport,sizeof(*d.hostReport),cudaMemcpyDeviceToHost,d.stream));
    checked(cudaStreamSynchronize(d.stream));
    const auto result=d.hostReport->state;
    const auto& stamps=d.hostReport->stamps;
    if(stamps[0]&&stamps[4]>=stamps[3]&&stamps[3]>=stamps[2]&&stamps[2]>=stamps[0]){
        std::fprintf(stderr,"THERMAL_TIMING sequence=%llu seconds=%.9f radiationAndInputs=%.9f interface=%.9f solidParticlesAndCommit=%.9f\n",
            static_cast<unsigned long long>(result.sequence),(stamps[4]-stamps[0])*1e-9,
            (stamps[2]-stamps[0])*1e-9,(stamps[3]-stamps[2])*1e-9,(stamps[4]-stamps[3])*1e-9);
    }
    return result;
}
}}
