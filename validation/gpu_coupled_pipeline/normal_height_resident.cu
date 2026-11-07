#include "gpu/NormalHeightMotion.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
namespace g=ugkp::gpu::geometry;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(const std::vector<T>& data){if(!data.empty()){
        check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};
__global__ void selectHeightInput(const g::NormalHeightMotionInput* seed,g::NormalHeightMotionInput* input){*input=*seed;}
}
std::vector<g::MotionPenalty> residentNormalHeight(const g::Topology& topology,int count,const std::vector<int>& owner,
    const std::vector<int>& walls,const std::vector<g::Point>& reference,const std::vector<g::Point>& points,
    const std::vector<g::Point>& displacement,const std::vector<unsigned char>& fixed,double tolerance,g::NormalHeightMotionControl& control){
    g::ResidentNormalHeightMotion solver(topology,count,owner,walls);
    Device<g::Point> ref(reference),p(points),delta(displacement);Device<unsigned char> mask(fixed);
    Device<g::NormalHeightMotionInput> seed(std::vector<g::NormalHeightMotionInput>{{ref.p,p.p,delta.p,mask.p,tolerance}}),
        input(std::vector<g::NormalHeightMotionInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    selectHeightInput<<<1,1,0,stream>>>(seed.p,input.p);solver.evaluate(input.p,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t size=0;check(cudaGraphGetNodes(graph,nullptr,&size));
    std::vector<cudaGraphNode_t> nodes(size);check(cudaGraphGetNodes(graph,nodes.data(),&size));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("normal-height graph contains a host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    const auto view=solver.view();std::vector<g::MotionPenalty> result(count);
    if(count)check(cudaMemcpy(result.data(),view.penalty,result.size()*sizeof(result[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&control,view.control,sizeof(control),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}

namespace {
__global__ void selectPointMotionInput(const g::NormalHeightPointMotionInput* seed,g::NormalHeightPointMotionInput* input){*input=*seed;}
}
g::NormalHeightPointMotionResult residentNormalHeightPointMotion(const g::NormalHeightPointMotionDefinition& definition,
    const std::vector<g::Point>& displacement){
    g::NormalHeightPointMotion solver(definition);
    Device<g::Point> reference(definition.reference),points(definition.points),boundary(displacement);
    Device<g::NormalHeightPointMotionInput> seed(std::vector<g::NormalHeightPointMotionInput>{{reference.p,points.p,boundary.p,definition.geometricTolerance}}),
        input(std::vector<g::NormalHeightPointMotionInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    selectPointMotionInput<<<1,1,0,stream>>>(seed.p,input.p);const auto view=solver.solveDevice(input.p,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("composed normal-height graph contains host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    g::NormalHeightPointMotionResult result;g::HarmonicDeviceStatus status;
    check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.constraints,solver.constraints().control,sizeof(result.constraints),cudaMemcpyDeviceToHost));
    result.motion.iterations=status.iterations;result.motion.residual=status.residual;
    result.motion.errorPoint=status.errorPoint;result.motion.errorKind=status.errorKind;
    if(status.errorPoint<0){result.motion.points.resize(definition.points.size());
        if(!result.motion.points.empty())check(cudaMemcpy(result.motion.points.data(),view.points,result.motion.points.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
