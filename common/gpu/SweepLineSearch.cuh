// Included by PolyhedralGeometry.cu.
#include "SweepLineSearch.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void copySweepSearchProblem(const SweepLineSearchProblem* input,SweepLineSearchProblem* output){*output=*input;}
__global__ void initializeSweepSearch(const SweepLineSearchProblem* input,SweepLineSearchProblem* operation,
    SweepLineSearchStatus* state,bool restart,cudaGraphConditionalHandle handle){
    if(restart){*operation=*input;*state={};}
    else {state->trials=0;state->candidate=false;state->evaluated=false;}
    const int limit=operation->maximumTrials>0?operation->maximumTrials:1;
    state->exhausted=state->nextBacktrack>=limit||::ldexp(1.,-state->nextBacktrack)==0;
    cudaGraphSetConditional(handle,!state->exhausted);
}
__global__ void bindSweepSearchTrial(const SweepLineSearchProblem* operation,const SweepLineSearchStatus* state,
    const Point* endpoint,SweepTrialProblem* trial,SweepResidualProblem* residual){
    *trial={operation->origin,operation->correction,state->nextBacktrack};
    *residual=operation->residual;residual->endpoint=endpoint;
}
__global__ void advanceSweepSearch(const SweepLineSearchProblem* operation,SweepLineSearchStatus* state,
    SweepTrialView trial,const SweepResidualStatus* residual,cudaGraphConditionalHandle handle){
    const auto coordinate=*trial.status;++state->nextBacktrack;
    if(coordinate.representable){
        if(!coordinate.changed){state->exhausted=true;cudaGraphSetConditional(handle,0);return;}
        ++state->trials;state->evaluated=true;state->trial.coordinate=coordinate;state->trial.residual=*residual;
        state->trial.sufficientDecrease=residual->failure==SweepResidualFailure::None
            &&(residual->metric<=1||residual->metric<operation->initialMetric*(1-1e-4*coordinate.alpha));
        state->candidate=state->trial.sufficientDecrease;
    }
    const int limit=operation->maximumTrials>0?operation->maximumTrials:1;
    state->exhausted=state->nextBacktrack>=limit||::ldexp(1.,-state->nextBacktrack)==0;
    cudaGraphSetConditional(handle,!state->candidate&&!state->exhausted);
}
}
struct ResidentSweepLineSearch::Implementation {
    ResidentSweepResidual& residual;ResidentSweepTrial trial;
    Array<SweepLineSearchProblem> input,operation;Array<SweepLineSearchStatus> state;
    Array<SweepTrialProblem> trialInput;Array<SweepResidualProblem> residualInput;
    cudaStream_t capture=nullptr;cudaGraph_t graph=nullptr,continuation=nullptr;cudaGraphExec_t executable=nullptr,resumeExecutable=nullptr;
    Implementation(int points,ResidentSweepResidual& owner):residual(owner),trial(points){
        input.reserve(1);operation.reserve(1);state.reserve(1);trialInput.reserve(1);residualInput.reserve(1);
        checked(cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking));
    }
    ~Implementation(){if(capture)cudaStreamSynchronize(capture);
        if(executable)cudaGraphExecDestroy(executable);if(resumeExecutable)cudaGraphExecDestroy(resumeExecutable);
        if(graph)cudaGraphDestroy(graph);if(continuation)cudaGraphDestroy(continuation);if(capture)cudaStreamDestroy(capture);}
};
ResidentSweepLineSearch::ResidentSweepLineSearch(int points,ResidentSweepResidual& owner):data_(new Implementation(points,owner)){
    auto& d=*data_;checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph,nullptr,d.input.pointer,true);
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
    checked(cudaGraphCreate(&d.continuation,0));appendTo(d.continuation,nullptr,d.input.pointer,false);
    checked(cudaGraphInstantiateWithFlags(&d.resumeExecutable,d.continuation,0));
}
ResidentSweepLineSearch::~ResidentSweepLineSearch()=default;
SweepLineSearchView ResidentSweepLineSearch::view()const{const auto& d=*data_;return {d.trial.view().points,d.state.pointer};}
ExecutionGraphNode ResidentSweepLineSearch::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,const SweepLineSearchProblem* problem,bool restart){
    auto& d=*data_;using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    auto previous=appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.capture,[&]{
        initializeSweepSearch<<<1,1,0,d.capture>>>(problem,d.operation.pointer,d.state.pointer,restart,handle);
    });
    cudaGraph_t body=nullptr;previous=addWhile(graph,handle,{previous},body);
    appendCaptured(body,{},d.capture,[&]{
        bindSweepSearchTrial<<<1,1,0,d.capture>>>(d.operation.pointer,d.state.pointer,d.trial.view().points,d.trialInput.pointer,d.residualInput.pointer);
        d.trial.evaluate(d.trialInput.pointer,d.capture);d.residual.evaluate(d.residualInput.pointer,d.capture);
        advanceSweepSearch<<<1,1,0,d.capture>>>(d.operation.pointer,d.state.pointer,d.trial.view(),d.residual.view().status,handle);
    });return previous;
}
void ResidentSweepLineSearch::begin(const SweepLineSearchProblem* problem,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copySweepSearchProblem<<<1,1,0,stream>>>(problem,d.input.pointer);checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,stream));
}
void ResidentSweepLineSearch::resume(void* rawStream){checked(cudaGraphLaunch(data_->resumeExecutable,static_cast<cudaStream_t>(rawStream)));}
}}}
