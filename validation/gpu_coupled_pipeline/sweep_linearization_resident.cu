#include "sweep_linearization_resident.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
namespace g=ugkp::gpu::geometry;
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(std::size_t size){if(size)check(cudaMalloc(&p,size*sizeof(T)));}
    explicit Device(const std::vector<T>& input):Device(input.size()){
        if(!input.empty())check(cudaMemcpy(p,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
    std::vector<T> read(std::size_t size)const{std::vector<T> out(size);
        if(size)check(cudaMemcpy(out.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost));return out;}
};
__global__ void selectDescriptor(const g::SweepLinearizationDeviceInput* inputs,int index,
    g::SweepLinearizationDeviceInput* selected){*selected=inputs[index];}
__global__ void recordColumn(int rows,const double* in,double* out){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)out[i]=in[i];}
}
ResidentSweepProbeResult residentSweepProbe(const g::Topology& topology,
    const std::vector<g::Point>& old,const std::vector<g::Point>& endpoint,
    const std::vector<int>& faces,const std::vector<double>& scales,
    const std::vector<g::PlanarConstraint>& planar,
    const std::vector<std::vector<g::Point>>& modes,const std::vector<double>& coefficients){
    g::ResidentSweepLinearization owner(topology,old.size(),faces,planar.size());
    Device<g::Point> dOld(old),dEnd(endpoint);Device<double> dScales(scales);
    Device<g::PlanarConstraint> dPlanar(planar);std::vector<g::Point> flat;
    for(const auto& mode:modes)flat.insert(flat.end(),mode.begin(),mode.end());
    Device<g::Point> dModes(flat);std::vector<g::SweepLinearizationDeviceInput> input;
    for(std::size_t i=0;i<modes.size();++i)input.push_back({dOld.p,dEnd.p,
        old.empty()?nullptr:dModes.p+i*old.size(),dScales.p,dPlanar.p,coefficients[i]});
    // Empty meshes still exercise descriptor selection, reset and graph reuse.
    if(input.empty())input.push_back({dOld.p,dEnd.p,nullptr,dScales.p,dPlanar.p,0});
    Device<g::SweepLinearizationDeviceInput> descriptors(input),selected(1);
    const int rows=faces.size()+3*planar.size();Device<double> columns(std::size_t(rows)*modes.size());
    const auto view=owner.view();cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    selectDescriptor<<<1,1,0,stream>>>(descriptors.p,0,selected.p);
    owner.planarity(selected.p,stream);owner.resetCorrection(stream);
    for(std::size_t i=0;i<modes.size();++i){
        selectDescriptor<<<1,1,0,stream>>>(descriptors.p,i,selected.p);
        owner.column(selected.p,stream);
        if(rows)recordColumn<<<(rows-1)/128+1,128,0,stream>>>(rows,view.column,columns.p+i*rows);
        owner.accumulate(selected.p,stream);
    }
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("resident sweep graph contains a host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
    check(cudaStreamSynchronize(stream));ResidentSweepProbeResult result;
    result.polynomials.resize(planar.size());result.correction.resize(old.size());
    if(!planar.empty())check(cudaMemcpy(result.polynomials.data(),view.polynomials,planar.size()*sizeof(g::Polynomial),cudaMemcpyDeviceToHost));
    if(!old.empty())check(cudaMemcpy(result.correction.data(),view.correction,old.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));
    const auto all=columns.read(std::size_t(rows)*modes.size());
    for(std::size_t i=0;i<modes.size();++i)result.columns.emplace_back(all.begin()+i*rows,all.begin()+(i+1)*rows);
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
