#include <cstdio>
#include "SolidThermalProperties.H"
#include "ImplicitDiffusion.H"
#include "ExecutionGraph.cuh"
#include "../Diagnostics.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA solid iteration: ")+cudaGetErrorString(code));}
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
    else if(!report->converged)recordWarning(context,makeRecord(Code::AccuracyResidual,34013,0,report->iterations,
        EntityKind::None,-1,-1,report->finalResidual,report->initialResidual,0,Quantity::LinearSystem));
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
        &&fmaxf(value.surfaceChange,value.surfaceFluxChange)<=controls.relativeTolerance&&residual.magnitude().lessEqual(limit);
    const bool again=state->valid&&!state->converged&&state->iteration<(controls.maximumIterations>1?controls.maximumIterations:1);
    cudaGraphSetConditional(handle,again);
    if(!again)printf("FP32_SOLID_STOP iterations=%d max=%d dT=%.9g dSurface=%.9g fluxMismatch=%.9g energyResidual=%.17g energyLimit=%.17g effectiveRelativeTolerance=%.9g effectiveEnergyRelativeTolerance=%.9g valid=%d converged=%d\n",
        state->iteration,controls.maximumIterations,double(value.temperatureChange),double(value.surfaceChange),
        double(value.surfaceFluxChange),residual.binary64(),limit.binary64(),double(controls.relativeTolerance),
        double(controls.energyRelativeTolerance),int(state->valid),int(state->converged));
    if(state->valid&&!state->converged&&!again){using namespace diagnostics;
        if(!residual.magnitude().lessEqual(limit))recordWarning(context,makeRecord(Code::ConservationResidual,34014,0,state->iteration,EntityKind::None,-1,-1,
            residual.binary64(),limit.binary64(),0,Quantity::Conservation));
        const GpuReal change=fmaxf(value.temperatureChange,fmaxf(value.surfaceChange,value.surfaceFluxChange));
        if(!(change<=controls.relativeTolerance))recordWarning(context,makeRecord(Code::AccuracyResidual,34015,0,state->iteration,EntityKind::None,-1,-1,
            change,controls.relativeTolerance,0,Quantity::Accuracy));
    }
}
}
struct SolidThermalIterationGraph::Implementation {
    cudaStream_t stream=nullptr;cudaEvent_t ready=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    SolidThermalIteration* report=nullptr;diagnostics::DeviceContext* diagnostics=nullptr;
    bool ownsDiagnostics=false,resultReady=false;
    std::vector<cudaStream_t> owners;
    SolidThermalProperties* properties=nullptr;ImplicitDiffusion* diffusion=nullptr;SparseConjugateGradient* sparse=nullptr;
    SolidThermalResidentProblem problem;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));checked(cudaMalloc(reinterpret_cast<void**>(&report),sizeof(*report)));}
    ~Implementation(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(ownsDiagnostics&&diagnostics)cudaFree(diagnostics);if(report)cudaFree(report);if(ready)cudaEventDestroy(ready);if(stream)cudaStreamDestroy(stream);}
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
    checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph);
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
ExecutionGraphNode SolidThermalIterationGraph::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,bool contactCandidate){
    auto& d=*data_;auto& properties=*d.properties;auto& diffusion=*d.diffusion;auto& sparse=*d.sparse;const auto& problem=d.problem;
    using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    cudaGraphConditionalHandle handle;
    checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    const auto initialized=appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        if(contactCandidate)diffusion.enqueueResidentWindowRestore(d.stream);
        diffusion.enqueueResidentInterfaceEnergy(d.stream);
        properties.enqueueResidentAnchors(diffusion.deviceTemperature(),contactCandidate?diffusion.deviceBoundaryTemperature():nullptr,problem.cells,d.stream);
        beginThermalWindow<<<1,1,0,d.stream>>>(d.report,problem.controls,d.diagnostics,handle,problem.interval,problem.deviceInterval);
        if(!contactCandidate)diffusion.enqueueBoundaryJumpSchedule(d.stream,&d.report->interval);
    });
    cudaGraph_t body=nullptr;const auto loop=addWhile(graph,handle,{initialized},body);(void)loop;
    const auto assembled=appendCaptured(body,{},d.stream,[&]{
        beginThermalIteration<<<1,1,0,d.stream>>>(d.report);diffusion.enqueueResidentSnapshot(d.stream);
        properties.enqueueResidentEvaluation(problem.coefficientPlan,diffusion.deviceTemperature(),nullptr,diffusion.deviceBoundaryTemperature(),d.stream,d.report);
        properties.enqueueResidentDiagnostic(d.diagnostics,d.stream);
        diffusion.enqueueResidentAssembly(d.stream,&d.report->interval);
    });
    const auto solved=sparse.appendResidentLocalLdu(body,assembled,diffusion.deviceSolution());
    appendCaptured(body,{solved},d.stream,[&]{
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
    });
    return loop;
}
SolidThermalIterationGraph::~SolidThermalIterationGraph()=default;
void SolidThermalIterationGraph::enqueue(void* producer){
    auto& d=*data_;d.waitFor(producer);checked(cudaGraphLaunch(d.executable,d.stream));d.resultReady=true;
    checked(cudaEventRecord(d.ready,d.stream));checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(producer),d.ready,0));
    for(auto stream:d.owners)checked(cudaStreamWaitEvent(stream,d.ready,0));
}
const SolidThermalIteration* SolidThermalIterationGraph::deviceReport()const{return data_->report;}
void* SolidThermalIterationGraph::producerStream()const{return data_->stream;}
SolidThermalIteration SolidThermalIterationGraph::collect(diagnostics::DeviceContext* diagnosticOutput){
    auto& d=*data_;if(!d.resultReady)throw std::logic_error("resident solid iteration output requested before enqueue");
    SolidThermalIteration result;checked(cudaMemcpyAsync(&result,d.report,sizeof(result),cudaMemcpyDeviceToHost,d.stream));
    if(diagnosticOutput)checked(cudaMemcpyAsync(diagnosticOutput,d.diagnostics,sizeof(*diagnosticOutput),cudaMemcpyDeviceToHost,d.stream));
    checked(cudaStreamSynchronize(d.stream));return result;
}

void SolidThermalIterationGraph::prepareContactWindow(const std::vector<int>& cells){
    auto& d=*data_;d.diffusion->prepareResidentWindowSnapshot();d.properties->prepareResidentContact(cells);
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
    const GpuReal energy=metrics->energyDifference/metrics->energyScale;
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
    cudaStream_t stream=nullptr;cudaEvent_t ready=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    SolidThermalIterationGraph *primary=nullptr,*auxiliary=nullptr;SolidThermalContact* contact=nullptr;
    SolidContactIteration* report=nullptr;bool resultReady=false;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));checked(cudaMalloc(reinterpret_cast<void**>(&report),sizeof(*report)));}
    ~Implementation(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        if(report)cudaFree(report);if(ready)cudaEventDestroy(ready);if(stream)cudaStreamDestroy(stream);}
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
    const auto initialized=appendCaptured(d.graph,{},d.stream,[&]{
        primary.enqueueContactWindowStart(d.stream);auxiliary.enqueueContactWindowStart(d.stream);contact.enqueueWindowStart(d.stream);
        beginContactWindow<<<1,1,0,d.stream>>>(d.report,context,handle);
    });
    cudaGraph_t body=nullptr;addWhile(d.graph,handle,{initialized},body);
    const auto exchange=appendCaptured(body,{},d.stream,[&]{contact.enqueueEvaluate(primary.contactState(),auxiliary.contactState(),interval,deviceInterval,d.stream);});
    // Both candidates consume the same contact-energy iterate and the accepted
    // window anchors; neither region consumes the other's unfinished candidate.
    const auto primaryDone=primary.appendTo(body,exchange,true);
    const auto auxiliaryDone=auxiliary.appendTo(body,exchange,true);
    appendCaptured(body,{primaryDone,auxiliaryDone},d.stream,[&]{
        primary.enqueueContactState(d.stream);auxiliary.enqueueContactState(d.stream);
        const auto metrics=contact.enqueueConvergence(primary.contactState(),auxiliary.contactState(),d.stream);
        finishContactIteration<<<1,1,0,d.stream>>>(d.report,metrics,controls,context,handle);
    });
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
SolidContactIterationGraph::~SolidContactIterationGraph()=default;
void SolidContactIterationGraph::enqueue(void* producerStream){
    auto& d=*data_;d.waitFor(producerStream);checked(cudaGraphLaunch(d.executable,d.stream));d.resultReady=true;
    d.primary->acceptExternalExecution(d.stream);d.auxiliary->acceptExternalExecution(d.stream);
    checked(cudaEventRecord(d.ready,d.stream));checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(producerStream),d.ready,0));
    checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(d.contact->primaryEnergy().producerStream),d.ready,0));
}
SolidContactIteration SolidContactIterationGraph::collect(){
    auto& d=*data_;if(!d.resultReady)throw std::logic_error("solid contact output requested before enqueue");
    SolidContactIteration result;checked(cudaMemcpyAsync(&result,d.report,sizeof(result),cudaMemcpyDeviceToHost,d.stream));
    checked(cudaStreamSynchronize(d.stream));return result;
}
}}
