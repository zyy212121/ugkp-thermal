#include "sweep_trial_owner_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {T* p=nullptr;explicit Device(const std::vector<T>& a){if(!a.empty()){check(cudaMalloc(&p,a.size()*sizeof(T)));check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}}~Device(){if(p)cudaFree(p);}};
__global__ void bindTrial(const g::SweepTrialProblem* source,g::SweepTrialProblem* output){*output=*source;}
}
SweepTrialOwnerProbe probeSweepTrialOwner(g::ResidentSweepTrial& owner,const std::vector<g::Point>& origin,const std::vector<g::Point>& correction,int backtrack){
    Device<g::Point> old(origin),delta(correction);Device<g::SweepTrialProblem> seed(std::vector<g::SweepTrialProblem>{{old.p,delta.p,backtrack}}),input(std::vector<g::SweepTrialProblem>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));bindTrial<<<1,1,0,stream>>>(seed.p,input.p);owner.evaluate(input.p,stream);check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("sweep trial graph has host communication");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));for(int repeat=0;repeat<3;++repeat)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    SweepTrialOwnerProbe result;result.points.resize(origin.size());const auto view=owner.view();
    check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    if(!origin.empty())check(cudaMemcpy(result.points.data(),view.points,origin.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
