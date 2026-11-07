#include "interface_motion_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& data){if(!data.empty()){check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};
__global__ void selectInterfaceMotion(const g::InterfaceGasPointMotionInput* seed,g::InterfaceGasPointMotionInput* input){*input=*seed;}
}
g::InterfaceGasPointMotionResult probeInterfaceMotion(g::InterfaceGasPointMotion& owner,const g::InterfaceGasPointMotionDefinition& d,
    const std::vector<g::Point>& endpoint,const std::vector<g::Point>& areas){
    Device<g::Point> old(d.oldSolid),end(endpoint),area(areas),gas(d.gas.points),reference(d.gas.reference);
    Device<double> depth(d.pointThickness),previous(d.previousFilmThickness),film(d.filmThickness);
    Device<g::InterfaceGasPointMotionInput> seed(std::vector<g::InterfaceGasPointMotionInput>{{
        {old.p,end.p,area.p,gas.p,{depth.p,sizeof(double)},{previous.p,sizeof(double)},{film.p,sizeof(double)}},reference.p,d.gas.geometricTolerance}}),
        input(std::vector<g::InterfaceGasPointMotionInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;cudaGraph_t child;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaGraphCreate(&graph,0));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));selectInterfaceMotion<<<1,1,0,stream>>>(seed.p,input.p);check(cudaStreamEndCapture(stream,&child));
    cudaGraphNode_t first;check(cudaGraphAddChildGraphNode(&first,graph,nullptr,0,child));check(cudaGraphDestroy(child));
    owner.appendTo(graph,first,input.p);check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    static bool exported=false;if(!exported){check(cudaGraphDebugDotPrint(graph,"interface-motion-graph.dot",cudaGraphDebugDotFlagsVerbose));exported=true;}
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
    // Also exercise the standalone graph after the externally composed graph.
    owner.solveDevice(input.p,stream);check(cudaStreamSynchronize(stream));
    const auto view=owner.view();g::InterfaceGasPointMotionResult result;
    check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    if(result.status.failure==g::InterfaceGasMotionFailure::None){result.points.resize(d.gas.points.size());
        if(!result.points.empty())check(cudaMemcpy(result.points.data(),view.points,result.points.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
