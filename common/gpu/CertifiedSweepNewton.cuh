
#pragma once
namespace ugkp { namespace gpu { namespace geometry {
namespace {
struct NewtonControl {
    SweepNewtonReport result;
    SweepResidualStatus retained;
    int restart=1,accepted=0;
};
__global__ void startSweepNewton(NewtonControl* state,const SweepResidualStatus* initial,
    cudaGraphConditionalHandle outer) {
    *state={};state->retained=*initial;
    cudaGraphSetConditional(outer,initial->failure==SweepResidualFailure::None&&initial->metric>1);
}
__global__ void startNewtonTrials(NewtonControl* state,HarmonicSweepDeviceView correction,
    double tolerance,cudaGraphConditionalHandle trial) {
    const auto solved=*correction.status;
    auto& r=state->result;
    r.modeSolves+=solved.modeSolves;
    if(solved.linearInitialized)r.rank=solved.rank;
    r.correctionError=solved.error;r.index=solved.errorPoint;
    for(int i=0;i<solved.modeSolves;++i){
        const auto mode=correction.modes[i];
        if(mode.errorPoint<0&&!(mode.residual<=tolerance)){
            ++r.modeWarnings;
            if(mode.residual>r.modeResidual)r.modeResidual=mode.residual;
        }
    }
    if(solved.linearValid){
        const auto limit=state->retained.consistencyTolerance;
        for(int i=r.rank;i<solved.rows;++i){
            const double value=correction.linear.rhs[i];
            if(::fabs(value)>limit){
                if(!r.rankWarnings||::fabs(value)>::fabs(r.rankResidual)){
                    r.rankResidual=value;r.rankTolerance=limit;r.rankRow=i;
                }
                ++r.rankWarnings;
            }
        }
    }
    state->accepted=0;state->restart=1;
    cudaGraphSetConditional(trial,solved.error==0);
}
__global__ void inspectNewtonTrial(NewtonControl* state,const SweepLineSearchStatus* search,
    cudaGraphConditionalHandle certify) {
    state->result.lineSearchTrials+=search->trials;
    state->restart=0;
    if(search->candidate)state->result.certificateCurrent=false;
    cudaGraphSetConditional(certify,search->candidate);
}
__device__ bool fatalSweepTrial(const diagnostics::DeviceContext* context) {
    if(!context||!context->fatalClaim)return false;
    const auto code=context->firstFatal.code;
    return code==diagnostics::Code::InvalidIndex||code==diagnostics::Code::MissingOperator||code==diagnostics::Code::CudaRuntime;
}
__global__ void allowGasCertificate(LocalFrameStageView solid,cudaGraphConditionalHandle allowed){
    cudaGraphSetConditional(allowed,solid.geometry.status->failure==StageGeometryFailure::None&&!solid.diagnostics->fatalClaim);
}
__global__ void certifyNewtonTrial(NewtonControl* state,const SweepLineSearchStatus* search,
    LocalFrameStageView solid,InterfaceGasStageView gas,const int* invalidWet) {
    auto& r=state->result;
    if(fatalSweepTrial(solid.diagnostics)){r.fatal=solid.diagnostics->firstFatal;return;}
    if(solid.geometry.status->failure!=StageGeometryFailure::None||solid.diagnostics->fatalClaim)return;
    if(gas.geometry.status){
        if(fatalSweepTrial(gas.diagnostics)){r.fatal=gas.diagnostics->firstFatal;return;}
        if(gas.motion.status->failure!=InterfaceGasMotionFailure::None||
            gas.geometry.status->failure!=StageGeometryFailure::None||gas.diagnostics->fatalClaim)return;
        if(invalidWet&&*invalidWet!=INT_MAX){r.wetFace=*invalidWet;return;}
    }
    state->accepted=1;state->retained=search->trial.residual;
    ++r.iterations;r.certificateCurrent=true;
}
__global__ void saveCertifiedPoints(const NewtonControl* state,int points,const Point* trial,Point* retained){
    if(!state->accepted)return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<points;i+=blockDim.x*gridDim.x)retained[i]=trial[i];
}
__global__ void continueNewtonTrial(const NewtonControl* state,const SweepLineSearchStatus* search,cudaGraphConditionalHandle trial){
    cudaGraphSetConditional(trial,!state->accepted&&!state->result.fatal.sourceSiteId&&state->result.wetFace<0
        &&search->candidate&&!search->exhausted);
}
__global__ void continueSweepNewton(NewtonControl* state,int maximum,cudaGraphConditionalHandle outer){
    state->result.retained=state->retained;
    cudaGraphSetConditional(outer,state->accepted&&state->retained.metric>1&&state->result.iterations<maximum);
}
}
SweepNewtonReport SweepResidualEvaluation::refine(HarmonicSweepCorrection& harmonic,
    LocalFrameStageGeometry& solid,InterfaceGasStage* gas,StageGeometryControls controls,
    double rankTolerance,double harmonicTolerance,int maximumIterations,int maximumTrials) {
    auto& d=*data_;
    const auto residual=view();const auto correction=harmonic.view();
    Array<NewtonControl> control;control.reserve(1);
    auto* state=control.pointer;
    struct Graph {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    } graph;
    checked(cudaGraphCreate(&graph.graph,0));
    const auto condition=[](cudaGraph_t parent){
        cudaGraphConditionalHandle flag;
        checked(cudaGraphConditionalHandleCreate(&flag,parent,0,cudaGraphCondAssignDefault));
        return flag;
    };
    using namespace executionGraphDetail;
    const auto outer=condition(graph.graph);
    auto start=appendCaptured(graph.graph,{},d.stream,[&]{startSweepNewton<<<1,1,0,d.stream>>>(state,residual.status,outer);});
    cudaGraph_t step=nullptr;addWhile(graph.graph,outer,{start},step);
    auto stepNode=harmonic.appendBorrowed(step,nullptr,d.endpoint.pointer,residual.rhs,
        &state->retained.consistencyTolerance,rankTolerance,planarity());
    const auto trial=condition(step);
    stepNode=appendCaptured(step,{stepNode},d.stream,[&]{
        startNewtonTrials<<<1,1,0,d.stream>>>(state,correction,harmonicTolerance,trial);
        bindSweepLineSearch<<<1,1,0,d.stream>>>(d.problem.pointer,d.endpoint.pointer,correction.correction,
            &state->retained.metric,maximumTrials,d.searchInput.pointer);
    });
    cudaGraph_t trials=nullptr;auto searched=addWhile(step,trial,{stepNode},trials);
    auto scan=d.search->appendTo(trials,nullptr,d.searchInput.pointer,true,&state->restart);
    const auto certify=condition(trials);cudaGraph_t certificate=nullptr;
    auto examined=appendCaptured(trials,{scan},d.stream,[&]{inspectNewtonTrial<<<1,1,0,d.stream>>>(state,d.search->view().status,certify);});
    auto validated=addIf(trials,certify,{examined},certificate);
    auto cert=solid.appendBorrowed(certificate,nullptr,d.search->view().points,controls);
    InterfaceGasStageView gasView{};const int* invalidWet=nullptr;
    if(gas){
        gasView=gas->view();invalidWet=gas->invalidWetOffset();
        const auto allowed=condition(certificate);
        const auto ready=appendCaptured(certificate,{cert},d.stream,[&]{allowGasCertificate<<<1,1,0,d.stream>>>(solid.view(),allowed);});
        cudaGraph_t gasBody=nullptr;cert=addIf(certificate,allowed,{ready},gasBody);
        gas->appendBorrowed(gasBody,nullptr,solid.view(),controls);
    }
    appendCaptured(certificate,{cert},d.stream,[&]{
        certifyNewtonTrial<<<1,1,0,d.stream>>>(state,d.search->view().status,solid.view(),gasView,invalidWet);
        if(d.points)saveCertifiedPoints<<<(d.points+127)/128,128,0,d.stream>>>(state,d.points,d.search->view().points,d.endpoint.pointer);
    });
    appendCaptured(trials,{validated},d.stream,[&]{continueNewtonTrial<<<1,1,0,d.stream>>>(state,d.search->view().status,trial);});
    appendCaptured(step,{searched},d.stream,[&]{continueSweepNewton<<<1,1,0,d.stream>>>(state,maximumIterations,outer);});
    checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
    checked(cudaGraphLaunch(graph.executable,d.stream));d.currentEndpoint=d.endpoint.pointer;
    SweepNewtonReport result{};
    d.transfer.download(&result,&state->result,sizeof(result));d.transfer.finish();
    return result;
}
}}}
