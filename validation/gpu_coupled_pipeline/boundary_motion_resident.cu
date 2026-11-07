#include "gpu/PolyhedralGeometry.H"
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
__global__ void selectMotion(const g::BoundaryMotionDeviceInput* initialized,g::BoundaryMotionDeviceInput* selected){*selected=*initialized;}
}
std::vector<g::BoundaryMotionPoint> residentBoundaryMotion(const g::Topology& topology,int count,
    const std::vector<g::Point>& areas,const std::vector<g::BoundaryMotionFace>& faces,
    const std::vector<unsigned char>& sides,double dt){
    std::vector<int> identities;for(const auto& face:faces)identities.push_back(face.face);
    g::ResidentBoundaryMotion owner(topology,count,identities,sides);
    Device<g::Point> a(areas);Device<g::BoundaryMotionFace> f(faces);
    Device<g::BoundaryMotionDeviceInput> initialized(std::vector<g::BoundaryMotionDeviceInput>{{a.p,f.p,dt}}),
        selected(std::vector<g::BoundaryMotionDeviceInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    selectMotion<<<1,1,0,stream>>>(initialized.p,selected.p);owner.evaluate(selected.p,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));
    std::vector<cudaGraphNode_t> list(nodes);check(cudaGraphGetNodes(graph,list.data(),&nodes));
    for(auto node:list){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("resident boundary motion graph has a host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
    check(cudaStreamSynchronize(stream));std::vector<g::BoundaryMotionPoint> out(count);
    if(count)check(cudaMemcpy(out.data(),owner.points(),out.size()*sizeof(out[0]),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return out;
}
