// Native resident-clock regression: header republication, graph reuse, changing
// intervals, frozen clocks, and rounding-edge timestamps. FP64 throughout.
#include <cuda_runtime.h>
#include "../../common/gasTransport/GasResidentInterval.H"
#include <cstdio>
#include <cstdlib>
#include <limits>
struct State { ugkwp::GasResidentInterval gasResidentInterval; };
static void checked(cudaError_t code){
    if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(1);}
}
int main(){
    using namespace ugkwp;
    State* state=nullptr;GasResidentInterval* clock=nullptr;
    checked(cudaMalloc(&state,sizeof(State)));checked(cudaMalloc(&clock,sizeof(GasResidentInterval)));
    cudaStream_t stream;checked(cudaStreamCreate(&stream));
    GasResidentInterval seed;seed.dt=0.125;seed.begin=0;seed.end=0.125;seed.middle=0.0625;seed.active=true;
    cudaGraph_t graph;cudaGraphExec_t exec;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    setResidentGasIntervalKernel<<<1,1,0,stream>>>(state,clock,seed,false);
    checked(cudaStreamEndCapture(stream,&graph));checked(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    size_t count=1;cudaGraphNode_t node;checked(cudaGraphGetNodes(graph,&node,&count));
    if(count!=1)return 2;
    cudaKernelNodeParams params{};checked(cudaGraphKernelNodeGetParams(node,&params));
    GasResidentInterval previous=seed;bool submitted=false,mode=false;int changes=0,continued=0;
    for(int i=0;i<2048;++i){
        GasResidentInterval next=previous;
        if(i==0)next=seed;
        else{
            next.begin=previous.end;
            if(i==512)next.dt=0.03;
            if(i==768)next.begin=0.0; // discontinuous restart clock
            if(i>=1024&&i<1152){next.begin=3.5;next.frozen=true;}
            else next.frozen=false;
            if(i==1280){next.begin=1.e20;next.dt=0.1;} // rounded begin==end
            if(i==1536){next.begin=-0.0;next.dt=std::numeric_limits<double>::denorm_min();}
            if(i==1792){next.begin=2.0;next.dt=0.125;}
            next.end=next.begin+next.dt;next.middle=next.begin+next.dt*0.5;
        }
        bool advance=submitted&&canAdvanceGasResidentInterval(previous,next);
        if(!submitted||advance!=mode||(!advance&&!sameGasResidentInterval(previous,next))){
            void* args[]={&state,&clock,&next,&advance};params.kernelParams=args;params.extra=nullptr;
            checked(cudaGraphExecKernelNodeSetParams(exec,node,&params));++changes;mode=advance;
        }
        // Capacity/geometry publication must not reset the graph-owned clock.
        checked(cudaMemsetAsync(state,0,sizeof(State),stream));
        checked(cudaGraphLaunch(exec,stream));
        State actual;GasResidentInterval actualClock;
        checked(cudaMemcpyAsync(&actual,state,sizeof(State),cudaMemcpyDeviceToHost,stream));
        checked(cudaMemcpyAsync(&actualClock,clock,sizeof(actualClock),cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
        if(!sameGasResidentInterval(actual.gasResidentInterval,next)||!sameGasResidentInterval(actualClock,next)){
            std::fprintf(stderr,"clock mismatch at interval %d\n",i);return 3;
        }
        continued+=advance;submitted=true;previous=next;
    }
    if(continued<1800||changes>32)return 4;
    checked(cudaGraphExecDestroy(exec));checked(cudaGraphDestroy(graph));
    checked(cudaFree(clock));checked(cudaFree(state));checked(cudaStreamDestroy(stream));
    std::printf("PASS 2048 exact intervals, %d resident advances, %d parameter updates\n",continued,changes);
}
