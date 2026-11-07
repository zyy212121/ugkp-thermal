#include "sweep_planarity_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {T* p=nullptr;explicit Device(const std::vector<T>& a){if(!a.empty()){check(cudaMalloc(&p,a.size()*sizeof(T)));check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}}~Device(){if(p)cudaFree(p);}};
__global__ void bindPlanarity(const g::SweepPlanarityInput* seed,g::SweepPlanarityInput* input){*input=*seed;}
}
SweepPlanarityProbe probeSweepPlanarity(g::ResidentSweepPlanarity& owner,const std::vector<g::Point>& points,double uncertainty){
    Device<g::Point> p(points);Device<g::SweepPlanarityInput> seed(std::vector<g::SweepPlanarityInput>{{p.p,uncertainty}}),input(std::vector<g::SweepPlanarityInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));bindPlanarity<<<1,1,0,stream>>>(seed.p,input.p);owner.evaluate(input.p,stream);check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset)throw std::runtime_error("planarity graph has host communication");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    const auto view=owner.view();int active=0;check(cudaMemcpy(&active,view.count,sizeof(active),cudaMemcpyDeviceToHost));
    if(active<0||active>owner.capacity())throw std::runtime_error("planarity count exceeds topology capacity");
    SweepPlanarityProbe result;result.planar.resize(active);result.tolerance.resize(active);
    if(active){check(cudaMemcpy(result.planar.data(),view.planar,active*sizeof(g::PlanarConstraint),cudaMemcpyDeviceToHost));check(cudaMemcpy(result.tolerance.data(),view.tolerance,active*sizeof(double),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
