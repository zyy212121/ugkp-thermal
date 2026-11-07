#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../../common/GpuCsrConditionalGraph.H"
static std::string lastError;
static void setLastError(const char* text,cudaError_t error){lastError=std::string(text)+": "+cudaGetErrorString(error);}
#include "../../common/GpuCsrConditionalGraph.cuh"
#include "../../common/gpu/ExecutionGraph.cuh"
struct State {
    State* deviceState=nullptr;
    ugkp::gpu::CsrAutomaticRuntime* csrAutomaticRuntime=nullptr;
    int csrHeavyReductionMode=2,csrHeavyReductionEnabled=1,csrHeavyReductionActive=1;
    cudaStream_t particleCaptureStream=nullptr;cudaGraph_t particleCaptureRoot=nullptr;
    void* cellScanTempStorage=nullptr;std::size_t cellScanTempBytes=0;
    int *csrCellTaskCount=nullptr,*csrCellTaskOffset=nullptr,*csrHeavyTaskCursor=nullptr;
    int *csrHeavyTaskCount=nullptr,*csrHeavyCellCount=nullptr;
    int nCells=1,csrHeavyWorkerGrid=1,multiprocessorCount=1;
    ugkp::gpu::CsrChoiceGraphs csrChoiceGraphs;
    int value=0,steps=0;
};
static void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}}
__global__ void reset(State* s){s->value=0;s->steps=0;}
__global__ void toggle(State* s){s->csrAutomaticRuntime->active=1-s->csrAutomaticRuntime->active;s->csrAutomaticRuntime->inspect=s->csrAutomaticRuntime->active;}
__global__ void add(State* s,int value){s->value+=value;}
__global__ void finish(State* s,cudaGraphConditionalHandle loop,int iterations){
    ++s->steps;s->value+=100;if(iterations>1)cudaGraphSetConditional(loop,s->steps<iterations);
}
int main(){
    using namespace ugkp::gpu::executionGraphDetail;
    int scenarios=0;
    for(int mode:{0,1,2})for(int iterations:{1,3})for(bool inspection:{false,true}){
        State host;host.csrHeavyReductionMode=mode;host.csrHeavyReductionEnabled=mode!=0;
        check(cudaMalloc(&host.deviceState,sizeof(State)));
        check(cudaMalloc(&host.csrAutomaticRuntime,sizeof(ugkp::gpu::CsrAutomaticRuntime)));
        ugkp::gpu::CsrAutomaticRuntime runtime;runtime.active=0;
        check(cudaMemcpy(host.csrAutomaticRuntime,&runtime,sizeof(runtime),cudaMemcpyHostToDevice));
        check(cudaMemcpy(host.deviceState,&host,sizeof(host),cudaMemcpyHostToDevice));
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaGraph_t graph;check(cudaGraphCreate(&graph,0));
        host.particleCaptureStream=stream;host.particleCaptureRoot=graph;
        cudaGraphConditionalHandle loop{};cudaGraph_t body=graph;
        if(iterations>1){
            auto start=appendCaptured(graph,{},stream,[&]{reset<<<1,1,0,stream>>>(host.deviceState);check(cudaGetLastError());});
            check(cudaGraphConditionalHandleCreate(&loop,graph,1,cudaGraphCondAssignDefault));
            addWhile(graph,loop,{start},body);
        }
        check(cudaStreamBeginCaptureToGraph(stream,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        if(iterations==1)reset<<<1,1,0,stream>>>(host.deviceState);
        toggle<<<1,1,0,stream>>>(host.deviceState);
        const int status=launchCsrConditional(&host,0,64,0,0,[&](cudaStream_t branch,bool heavy){
            add<<<1,1,0,branch>>>(host.deviceState,heavy?2:7);return int(cudaGetLastError()!=cudaSuccess);
        },inspection);
        if(status){std::fprintf(stderr,"%s\n",lastError.c_str());return 2;}
        finish<<<1,1,0,stream>>>(host.deviceState,loop,iterations);
        check(cudaGetLastError());check(cudaStreamEndCapture(stream,&body));
        host.particleCaptureStream=nullptr;host.particleCaptureRoot=nullptr;
        cudaGraphExec_t exec;check(cudaGraphInstantiateWithFlags(&exec,graph,0));
        int active=0;
        for(int repeat=0;repeat<20;++repeat){
            int expected=0;
            for(int i=0;i<iterations;++i){active=1-active;expected+=100+((mode==2&&inspection&&!active)?0:((mode==2?active:mode!=0)?2:7));}
            check(cudaGraphLaunch(exec,stream));State actual;
            check(cudaMemcpyAsync(&actual,host.deviceState,sizeof(actual),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(actual.value!=expected||actual.steps!=iterations){
                std::fprintf(stderr,"mode %d iterations %d repeat %d: %d != %d\n",mode,iterations,repeat,actual.value,expected);return 3;
            }
        }
        // The ordinary uncaptured owner remains a valid caller of the same API.
        reset<<<1,1>>>(host.deviceState);toggle<<<1,1>>>(host.deviceState);active=1-active;
        if(launchCsrConditional(&host,0,64,0,0,[&](cudaStream_t branch,bool heavy){
            add<<<1,1,0,branch>>>(host.deviceState,heavy?2:7);return int(cudaGetLastError()!=cudaSuccess);
        },inspection))return 4;
        State actual;check(cudaMemcpy(&actual,host.deviceState,sizeof(actual),cudaMemcpyDeviceToHost));
        if(actual.value!=((mode==2&&inspection&&!active)?0:((mode==2?active:mode!=0)?2:7)))return 5;
        ugkp::gpu::releaseCsrChoiceGraphs(host.csrChoiceGraphs);
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        check(cudaFree(host.csrAutomaticRuntime));check(cudaFree(host.deviceState));++scenarios;
    }
    std::printf("PASS %d direct/nested CSR contexts, 240 graph replays and twelve direct calls\n",scenarios);
}
