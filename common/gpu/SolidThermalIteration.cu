#include "SolidThermalProperties.H"
#include "ImplicitDiffusion.H"
#include "ExecutionGraph.cuh"
#include "GraphContinuationTask.cuh"
#include "../Diagnostics.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA solid iteration: ")+cudaGetErrorString(code));}
__global__ void thermalStageSelect(const int* stage,int expected,const int* pending,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,*stage==expected&&(!pending||!*pending));
}
__global__ void thermalInitialStage(int* stage,const SolidThermalIteration* report){if(!threadIdx.x)*stage=report->valid?1:3;}
__global__ void thermalStageSet(int* stage,int value){if(!threadIdx.x)*stage=value;}
__global__ void thermalResumeLoop(const int* stage,const int* pending,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,*stage!=3&&(!pending||!*pending));
}
template<class F> cudaGraphNode_t thermalStage(cudaGraph_t graph,cudaGraphNode_t dependency,cudaStream_t stream,
    int* stage,int expected,const int* pending,F capture){
    cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    auto selected=executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        thermalStageSelect<<<1,1,0,stream>>>(stage,expected,pending,handle);
    });
    cudaGraph_t body=nullptr;auto node=executionGraphDetail::addIf(graph,handle,std::vector<cudaGraphNode_t>{selected},body);capture(body);return node;
}
__global__ void beginThermalWindow(SolidThermalIteration* state,SolidThermalIterationControls controls,
    diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle,double interval,const double* deviceInterval){
    if(threadIdx.x)return;*state={};state->relaxation=controls.initialRelaxation;state->interval=deviceInterval?*deviceInterval:interval;
    state->valid=!diagnostics::hasFatal(context);cudaGraphSetConditional(handle,state->valid);
}
__global__ void beginThermalIteration(SolidThermalIteration* state){if(!threadIdx.x)++state->iteration;}
__global__ void checkThermalLinearReport(const ConjugateGradientReport* report,diagnostics::DeviceContext* context){
    using namespace diagnostics;if(threadIdx.x)return;
    if(!isfinite(report->finalResidual))recordFatal(context,makeRecord(Code::NonFinite,34012,0,report->iterations,
        EntityKind::None,-1,-1,report->finalResidual,report->initialResidual,0,Quantity::LinearSystem));
    else if(!report->converged){
        const int site=report->budgetExhausted?34013:(report->stagnated?34018:(report->singular?34019:34024));
        recordWarning(context,makeRecord(Code::AccuracyResidual,site,0,report->iterations,
            EntityKind::None,-1,-1,report->finalResidual,report->targetResidual,0,Quantity::LinearSystem));
    }
}
__global__ void finishThermalIteration(SolidThermalIteration* state,const SolidThermalConvergence* metrics,
    SolidThermalIterationControls controls,diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;state->convergence=*metrics;const auto& value=*metrics;
    auto residual=value.enthalpyChange;residual.merge(value.interfaceEnergy.negated());residual.merge(value.externalEnergy.negated());
    auto incoming=value.interfaceEnergy.magnitude();incoming.merge(value.externalEnergy.magnitude());
    auto scale=value.enthalpyChange.magnitude();if(scale.lessEqual(incoming))scale=incoming;
    ScaledThermalSum unity;unity.add(1);if(scale.lessEqual(unity))scale=unity;
    auto limit=scale.multiplied(controls.energyRelativeTolerance);limit.add(controls.energyAbsoluteTolerance);
    state->energyResidual=residual;state->energyLimit=limit;state->valid=!diagnostics::hasFatal(context);
    state->converged=state->valid&&value.temperatureChange<=controls.relativeTolerance
        &&fmax(value.surfaceChange,value.surfaceFluxChange)<=controls.relativeTolerance&&residual.magnitude().lessEqual(limit);
    const bool again=state->valid&&!state->converged&&state->iteration<(controls.maximumIterations>1?controls.maximumIterations:1);
    cudaGraphSetConditional(handle,again);
    if(state->valid&&!state->converged&&!again){using namespace diagnostics;
        if(!residual.magnitude().lessEqual(limit))recordWarning(context,makeRecord(Code::ConservationResidual,34014,0,state->iteration,EntityKind::None,-1,-1,
            residual.binary64(),limit.binary64(),0,Quantity::Conservation));
        const double change=fmax(value.temperatureChange,fmax(value.surfaceChange,value.surfaceFluxChange));
        if(!(change<=controls.relativeTolerance))recordWarning(context,makeRecord(Code::AccuracyResidual,34015,0,state->iteration,EntityKind::None,-1,-1,
            change,controls.relativeTolerance,0,Quantity::Accuracy));
    }
}
__global__ void finishThermalStage(int* stage,const SolidThermalIteration* report,SolidThermalIterationControls controls){
    if(!threadIdx.x)*stage=report->valid&&!report->converged&&report->iteration<(controls.maximumIterations>1?controls.maximumIterations:1)?1:3;
}

}
struct SolidThermalIterationGraph::Implementation {
    GraphContinuationTask task;std::shared_future<void> externalCompletion;
    cudaStream_t stream=nullptr;cudaEvent_t ready=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    SolidThermalIteration* report=nullptr;diagnostics::DeviceContext* diagnostics=nullptr;
    bool ownsDiagnostics=false,resultReady=false;int* stage=nullptr;
    std::vector<cudaStream_t> owners;
    SolidThermalProperties* properties=nullptr;ImplicitDiffusion* diffusion=nullptr;SparseConjugateGradient* sparse=nullptr;
    SolidThermalResidentProblem problem;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));checked(cudaMalloc(reinterpret_cast<void**>(&report),sizeof(*report)));}
    ~Implementation(){if(externalCompletion.valid())externalCompletion.wait();task.reset();if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(stage)cudaFree(stage);if(ownsDiagnostics&&diagnostics)cudaFree(diagnostics);if(report)cudaFree(report);if(ready)cudaEventDestroy(ready);if(stream)cudaStreamDestroy(stream);}
    void waitFor(void* producer){checked(cudaEventRecord(ready,static_cast<cudaStream_t>(producer)));checked(cudaStreamWaitEvent(stream,ready,0));}
};
SolidThermalIterationGraph::SolidThermalIterationGraph(SolidThermalProperties& properties,ImplicitDiffusion& diffusion,
    SparseConjugateGradient& sparse,const SolidThermalResidentProblem& problem):data_(new Implementation){
    auto& d=*data_;d.diagnostics=problem.diagnostics;
    if(!d.diagnostics){d.ownsDiagnostics=true;checked(cudaMalloc(reinterpret_cast<void**>(&d.diagnostics),sizeof(*d.diagnostics)));
        checked(cudaMemsetAsync(d.diagnostics,0,sizeof(*d.diagnostics),d.stream));}
    properties.prepareResidentIteration(problem.cells);
    d.owners={static_cast<cudaStream_t>(properties.deviceFields().readyStream),static_cast<cudaStream_t>(diffusion.deviceValues().readyStream)};
    d.waitFor(properties.deviceFields().readyStream);d.waitFor(diffusion.deviceValues().readyStream);
    d.properties=&properties;d.diffusion=&diffusion;d.sparse=&sparse;d.problem=problem;
    checked(cudaMalloc(reinterpret_cast<void**>(&d.stage),sizeof(int)));checked(cudaMemsetAsync(d.stage,0,sizeof(int),d.stream));
    checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph);
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
ExecutionGraphNode SolidThermalIterationGraph::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,bool contactCandidate){
    auto& d=*data_;auto& properties=*d.properties;auto& diffusion=*d.diffusion;auto& sparse=*d.sparse;const auto& problem=d.problem;
    using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    cudaGraphConditionalHandle handle;
    checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    const auto initialized=thermalStage(graph,dependency,d.stream,d.stage,0,nullptr,[&](cudaGraph_t initGraph){
      appendCaptured(initGraph,{},d.stream,[&]{
        if(contactCandidate)diffusion.enqueueResidentWindowRestore(d.stream);
        diffusion.enqueueResidentInterfaceEnergy(d.stream);
        properties.enqueueResidentAnchors(diffusion.deviceTemperature(),contactCandidate?diffusion.deviceBoundaryTemperature():nullptr,problem.cells,d.stream);
        beginThermalWindow<<<1,1,0,d.stream>>>(d.report,problem.controls,d.diagnostics,handle,problem.interval,problem.deviceInterval);
        if(!contactCandidate)diffusion.enqueueBoundaryJumpSchedule(d.stream,&d.report->interval);
        thermalInitialStage<<<1,1,0,d.stream>>>(d.stage,d.report);
      });
    });
    cudaGraph_t body=nullptr;const auto resumed=appendCaptured(graph,{initialized},d.stream,[&]{thermalResumeLoop<<<1,1,0,d.stream>>>(d.stage,sparse.capacityPendingDevice(),handle);});
    const auto loop=addWhile(graph,handle,{resumed},body);(void)loop;
    const auto assembled=thermalStage(body,nullptr,d.stream,d.stage,1,sparse.capacityPendingDevice(),[&](cudaGraph_t assembly){
      appendCaptured(assembly,{},d.stream,[&]{
        beginThermalIteration<<<1,1,0,d.stream>>>(d.report);diffusion.enqueueResidentSnapshot(d.stream);
        properties.enqueueResidentEvaluation(problem.coefficientPlan,diffusion.deviceTemperature(),nullptr,diffusion.deviceBoundaryTemperature(),d.stream,d.report);
        properties.enqueueResidentDiagnostic(d.diagnostics,d.stream);
        diffusion.enqueueResidentAssembly(d.stream,&d.report->interval);
        thermalStageSet<<<1,1,0,d.stream>>>(d.stage,2);
      });
    });
    const auto solved=thermalStage(body,assembled,d.stream,d.stage,2,sparse.capacityPendingDevice(),[&](cudaGraph_t solveGraph){sparse.appendResidentLocalLdu(solveGraph,nullptr,diffusion.deviceSolution());});
    const auto finished=thermalStage(body,solved,d.stream,d.stage,2,sparse.capacityPendingDevice(),[&](cudaGraph_t finalGraph){
      appendCaptured(finalGraph,{},d.stream,[&]{
        checkThermalLinearReport<<<1,1,0,d.stream>>>(sparse.deviceReport(),d.diagnostics);
        if(problem.controls.aitken)properties.enqueueResidentRelaxation(diffusion.deviceSolution(),diffusion.devicePreviousSolution(),problem.cells,d.report,problem.controls,d.stream);
        diffusion.enqueueResidentBoundaryTemperature(d.stream);
        properties.enqueueResidentEvaluation(problem.convergencePlan,diffusion.deviceTemperature(),diffusion.devicePreviousSolution(),diffusion.deviceBoundaryTemperature(),d.stream);
        properties.enqueueResidentDiagnostic(d.diagnostics,d.stream);
        diffusion.enqueueResidentBoundaryFlux(d.stream);
        const auto metrics=properties.enqueueResidentConvergence(problem.cells,problem.surfaceProperties,diffusion.deviceVolume(),
            diffusion.deviceBoundaryFlux(),diffusion.deviceBoundaryTemperature(),diffusion.deviceBoundaryGradient(),diffusion.deviceBoundaryEnergy(),
            diffusion.deviceBoundaryArea(),problem.interval,d.stream,&d.report->interval);
        finishThermalIteration<<<1,1,0,d.stream>>>(d.report,metrics,problem.controls,d.diagnostics,handle);
        finishThermalStage<<<1,1,0,d.stream>>>(d.stage,d.report,problem.controls);
      });
    });
    appendCaptured(body,{finished},d.stream,[&]{thermalResumeLoop<<<1,1,0,d.stream>>>(d.stage,sparse.capacityPendingDevice(),handle);});
    return loop;
}
SolidThermalIterationGraph::~SolidThermalIterationGraph()=default;
void SolidThermalIterationGraph::resetContinuation(void* stream){thermalStageSet<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(data_->stage,0);}
const int* SolidThermalIterationGraph::continuationStage()const{return data_->stage;}
const int* SolidThermalIterationGraph::capacityPendingDevice()const{return data_->sparse->capacityPendingDevice();}
bool SolidThermalIterationGraph::servicePendingCapacity(){return data_->sparse->servicePendingCapacity();}
void SolidThermalIterationGraph::awaitCompletion()const{
    auto& d=*data_;d.task.wait();if(d.externalCompletion.valid())d.externalCompletion.get();
}
void SolidThermalIterationGraph::acceptExternalCompletion(const std::shared_future<void>& completion){data_->externalCompletion=completion;}
void SolidThermalIterationGraph::enqueue(void* producer){
    auto& d=*data_;awaitCompletion();d.externalCompletion={};d.waitFor(producer);resetContinuation(d.stream);d.resultReady=true;
    d.task.start(d.stream,d.executable,[this]{return servicePendingCapacity();},[this]{
        auto& state=*data_;checked(cudaEventRecord(state.ready,state.stream));
        for(auto stream:state.owners)checked(cudaStreamWaitEvent(stream,state.ready,0));
    });
}
const SolidThermalIteration* SolidThermalIterationGraph::deviceReport()const{awaitCompletion();return data_->report;}
void* SolidThermalIterationGraph::producerStream()const{awaitCompletion();return data_->stream;}
SolidThermalIteration SolidThermalIterationGraph::collect(diagnostics::DeviceContext* diagnosticOutput){
    awaitCompletion();auto& d=*data_;if(!d.resultReady)throw std::logic_error("resident solid iteration output requested before enqueue");
    SolidThermalIteration result;checked(cudaMemcpyAsync(&result,d.report,sizeof(result),cudaMemcpyDeviceToHost,d.stream));
    if(diagnosticOutput)checked(cudaMemcpyAsync(diagnosticOutput,d.diagnostics,sizeof(*diagnosticOutput),cudaMemcpyDeviceToHost,d.stream));
    checked(cudaStreamSynchronize(d.stream));return result;
}

void SolidThermalIterationGraph::prepareContactWindow(const std::vector<int>& cells){
    awaitCompletion();auto& d=*data_;d.diffusion->prepareResidentWindowSnapshot();d.properties->prepareResidentContact(cells);
    d.waitFor(d.properties->deviceFields().readyStream);
}
namespace {
__global__ void setThermalInterval(SolidThermalIteration* report,double interval,const double* deviceInterval){
    if(!threadIdx.x)report->interval=deviceInterval?*deviceInterval:interval;
}
__global__ void beginContactWindow(SolidContactIteration* report,diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;*report={};report->valid=!diagnostics::hasFatal(context);cudaGraphSetConditional(handle,report->valid);
}
__global__ void finishContactIteration(SolidContactIteration* report,const SolidContactConvergence* metrics,
    SolidContactIterationControls controls,diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;++report->iteration;report->convergence=*metrics;report->valid=!diagnostics::hasFatal(context);
    const double energy=metrics->energyDifference/metrics->energyScale;
    report->converged=report->valid&&metrics->temperatureChange<=controls.temperatureTolerance&&energy<=controls.energyTolerance;
    const bool again=report->valid&&!report->converged&&report->iteration<(controls.maximumIterations>1?controls.maximumIterations:1);
    cudaGraphSetConditional(handle,again);
    if(report->valid&&!report->converged&&!again){using namespace diagnostics;
        if(!(metrics->temperatureChange<=controls.temperatureTolerance))recordWarning(context,makeRecord(Code::AccuracyResidual,34016,0,report->iteration,
            EntityKind::None,-1,-1,metrics->temperatureChange,controls.temperatureTolerance,0,Quantity::Accuracy));
        if(!(energy<=controls.energyTolerance))recordWarning(context,makeRecord(Code::ConservationResidual,34017,0,report->iteration,
            EntityKind::None,-1,-1,energy,controls.energyTolerance,0,Quantity::Conservation));
    }
}
__global__ void contactResume(int* stage,const int* primaryPending,const int* auxiliaryPending,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,*stage!=3&&(!primaryPending||!*primaryPending)&&(!auxiliaryPending||!*auxiliaryPending));
}
__global__ void contactReady(const int* primary,const int* auxiliary,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,*primary==3&&*auxiliary==3);
}
__global__ void contactInitialStage(int* stage,const SolidContactIteration* report){if(!threadIdx.x)*stage=report->valid?1:3;}
__global__ void contactFinishedStage(int* stage,const SolidContactIteration* report,SolidContactIterationControls controls){
    if(!threadIdx.x)*stage=report->valid&&!report->converged&&report->iteration<(controls.maximumIterations>1?controls.maximumIterations:1)?1:3;
}

}
void SolidThermalIterationGraph::enqueueContactWindowStart(void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    d.diffusion->enqueueResidentWindowSnapshot(stream);
    setThermalInterval<<<1,1,0,stream>>>(d.report,d.problem.interval,d.problem.deviceInterval);
    d.diffusion->enqueueBoundaryJumpSchedule(stream,&d.report->interval);
    enqueueContactState(stream);
}
void SolidThermalIterationGraph::enqueueContactState(void* stream){
    auto& d=*data_;d.properties->enqueueResidentContact(d.diffusion->deviceTemperature(),d.diagnostics,stream);
}
ThermalSolidContactDeviceView SolidThermalIterationGraph::contactState()const{return data_->properties->residentContact();}
void SolidThermalIterationGraph::acceptExternalExecution(void* producerStream){
    auto& d=*data_;d.resultReady=true;
    checked(cudaEventRecord(d.ready,static_cast<cudaStream_t>(producerStream)));checked(cudaStreamWaitEvent(d.stream,d.ready,0));
    for(auto stream:d.owners)checked(cudaStreamWaitEvent(stream,d.ready,0));
}
diagnostics::DeviceContext* SolidThermalIterationGraph::diagnosticsContext()const{return data_->diagnostics;}
struct SolidContactIterationGraph::Implementation {
    GraphContinuationTask task;
    cudaStream_t stream=nullptr;cudaEvent_t ready=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    SolidThermalIterationGraph *primary=nullptr,*auxiliary=nullptr;SolidThermalContact* contact=nullptr;
    SolidContactIteration* report=nullptr;bool resultReady=false;int* stage=nullptr;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));checked(cudaMalloc(reinterpret_cast<void**>(&report),sizeof(*report)));}
    ~Implementation(){task.reset();if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(stage)cudaFree(stage);if(report)cudaFree(report);if(ready)cudaEventDestroy(ready);if(stream)cudaStreamDestroy(stream);}
    void waitFor(void* owner){checked(cudaEventRecord(ready,static_cast<cudaStream_t>(owner)));checked(cudaStreamWaitEvent(stream,ready,0));}
};
SolidContactIterationGraph::SolidContactIterationGraph(SolidThermalIterationGraph& primary,SolidThermalIterationGraph& auxiliary,
    SolidThermalContact& contact,const SolidContactIterationControls& controls,double interval,const double* deviceInterval):data_(new Implementation){
    auto& d=*data_;d.primary=&primary;d.auxiliary=&auxiliary;d.contact=&contact;
    auto* context=primary.diagnosticsContext();
    if(context!=auxiliary.diagnosticsContext())throw std::invalid_argument("coupled solid graphs require one shared diagnostic owner");
    d.waitFor(primary.producerStream());d.waitFor(auxiliary.producerStream());d.waitFor(contact.primaryEnergy().producerStream);
    checked(cudaGraphCreate(&d.graph,0));cudaGraphConditionalHandle handle;
    checked(cudaGraphConditionalHandleCreate(&handle,d.graph,0,cudaGraphCondAssignDefault));
    using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    checked(cudaMalloc(reinterpret_cast<void**>(&d.stage),sizeof(int)));checked(cudaMemsetAsync(d.stage,0,sizeof(int),d.stream));
    const auto initialized=thermalStage(d.graph,nullptr,d.stream,d.stage,0,nullptr,[&](cudaGraph_t initial){
      appendCaptured(initial,{},d.stream,[&]{
        primary.enqueueContactWindowStart(d.stream);auxiliary.enqueueContactWindowStart(d.stream);contact.enqueueWindowStart(d.stream);
        beginContactWindow<<<1,1,0,d.stream>>>(d.report,context,handle);
        contactInitialStage<<<1,1,0,d.stream>>>(d.stage,d.report);
      });
    });
    const auto resumed=appendCaptured(d.graph,{initialized},d.stream,[&]{
        contactResume<<<1,1,0,d.stream>>>(d.stage,primary.capacityPendingDevice(),auxiliary.capacityPendingDevice(),handle);
    });
    cudaGraph_t body=nullptr;addWhile(d.graph,handle,{resumed},body);
    const auto exchange=thermalStage(body,nullptr,d.stream,d.stage,1,nullptr,[&](cudaGraph_t exchangeGraph){
      appendCaptured(exchangeGraph,{},d.stream,[&]{
        contact.enqueueEvaluate(primary.contactState(),auxiliary.contactState(),interval,deviceInterval,d.stream);
        primary.resetContinuation(d.stream);auxiliary.resetContinuation(d.stream);
        thermalStageSet<<<1,1,0,d.stream>>>(d.stage,2);
      });
    });
    const auto primaryDone=primary.appendTo(body,exchange,true);
    const auto auxiliaryDone=auxiliary.appendTo(body,exchange,true);
    cudaGraphConditionalHandle ready;checked(cudaGraphConditionalHandleCreate(&ready,body,0,cudaGraphCondAssignDefault));
    const auto selected=appendCaptured(body,{primaryDone,auxiliaryDone},d.stream,[&]{
        contactReady<<<1,1,0,d.stream>>>(primary.continuationStage(),auxiliary.continuationStage(),ready);
    });
    cudaGraph_t finishedGraph=nullptr;const auto finished=executionGraphDetail::addIf(body,ready,{selected},finishedGraph);
    appendCaptured(finishedGraph,{},d.stream,[&]{
        primary.enqueueContactState(d.stream);auxiliary.enqueueContactState(d.stream);
        const auto metrics=contact.enqueueConvergence(primary.contactState(),auxiliary.contactState(),d.stream);
        finishContactIteration<<<1,1,0,d.stream>>>(d.report,metrics,controls,context,handle);
        contactFinishedStage<<<1,1,0,d.stream>>>(d.stage,d.report,controls);
    });
    appendCaptured(body,{finished},d.stream,[&]{
        contactResume<<<1,1,0,d.stream>>>(d.stage,primary.capacityPendingDevice(),auxiliary.capacityPendingDevice(),handle);
    });
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
SolidContactIterationGraph::~SolidContactIterationGraph()=default;
void SolidContactIterationGraph::awaitCompletion()const{data_->task.wait();}
void SolidContactIterationGraph::enqueue(void* producerStream){
    auto& d=*data_;awaitCompletion();d.primary->awaitCompletion();d.auxiliary->awaitCompletion();d.waitFor(producerStream);thermalStageSet<<<1,1,0,d.stream>>>(d.stage,0);d.resultReady=true;
    d.task.start(d.stream,d.executable,[this]{
        const bool first=data_->primary->servicePendingCapacity();
        const bool second=data_->auxiliary->servicePendingCapacity();return first||second;
    },[this]{
        auto& state=*data_;state.primary->acceptExternalExecution(state.stream);state.auxiliary->acceptExternalExecution(state.stream);
        checked(cudaEventRecord(state.ready,state.stream));
        checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(state.contact->primaryEnergy().producerStream),state.ready,0));
    });
    d.primary->acceptExternalCompletion(d.task.result());d.auxiliary->acceptExternalCompletion(d.task.result());
}
SolidContactIteration SolidContactIterationGraph::collect(){
    awaitCompletion();auto& d=*data_;if(!d.resultReady)throw std::logic_error("solid contact output requested before enqueue");
    SolidContactIteration result;checked(cudaMemcpyAsync(&result,d.report,sizeof(result),cudaMemcpyDeviceToHost,d.stream));
    checked(cudaStreamSynchronize(d.stream));return result;
}
}}
