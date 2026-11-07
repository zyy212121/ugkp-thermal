#include "gas_drive_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include "common/gasTransport/GasDriveSelection.cuh"
namespace gasDriveProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;
        check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
};
struct Input {chmt::ResidentMaterialDriveRecord held,sample;double tractionScale=1;};
__global__ void compare(int count,const Input* input,ugkwp::GasDriveComparison* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    output[i]=ugkwp::compareGasDrive<chmt::Ns>(input[i].held,input[i].sample,input[i].tractionScale);
}
std::vector<ugkwp::GasDriveComparison> evaluate(const std::vector<Pair>& pairs){
    std::vector<chmt::CoupledSurfaceDownload> faces;std::vector<chmt::CoupledWallDownload> walls;
    for(const auto& p:pairs){faces.insert(faces.end(),p.held.begin(),p.held.end());faces.insert(faces.end(),p.sample.begin(),p.sample.end());
        walls.insert(walls.end(),p.heldWall.begin(),p.heldWall.end());walls.insert(walls.end(),p.sampleWall.begin(),p.sampleWall.end());}
    Device<chmt::CoupledSurfaceDownload> df(faces);Device<chmt::CoupledWallDownload> dw(walls);
    std::vector<Input> inputs;std::size_t f=0,w=0;
    for(const auto& p:pairs){Input in;in.tractionScale=p.tractionScale;
        in.held={df.p?df.p+f:nullptr,dw.p?dw.p+w:nullptr,p.held.size(),p.heldWall.size()};f+=p.held.size();w+=p.heldWall.size();
        in.sample={df.p?df.p+f:nullptr,dw.p?dw.p+w:nullptr,p.sample.size(),p.sampleWall.size()};f+=p.sample.size();w+=p.sampleWall.size();inputs.push_back(in);}
    Device<Input> input(inputs);Device<ugkwp::GasDriveComparison> out(std::vector<ugkwp::GasDriveComparison>(pairs.size()));
    check(cudaDeviceSynchronize());
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    if(!pairs.empty())compare<<<(pairs.size()+127)/128,128,0,stream>>>(pairs.size(),input.p,out.p);
    check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("driving comparison graph contains a non-kernel node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(executable,stream));
    check(cudaStreamSynchronize(stream));std::vector<ugkwp::GasDriveComparison> result(pairs.size());
    if(!result.empty())check(cudaMemcpy(result.data(),out.p,result.size()*sizeof(*out.p),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
struct TestHistory {
    const Input* pair=nullptr;
    const chmt::ResidentMaterialDriveInterval* interval=nullptr;
    std::size_t frames=2;
    __host__ __device__ chmt::ResidentMaterialDriveRecord record(std::size_t i)const{return i?pair->sample:pair->held;}
};
std::vector<ugkwp::GasDriveSelection> select(const std::vector<Pair>& pairs,const std::vector<SelectionTest>& tests){
    std::vector<chmt::CoupledSurfaceDownload> faces;std::vector<chmt::CoupledWallDownload> walls;
    for(const auto& p:pairs){faces.insert(faces.end(),p.held.begin(),p.held.end());faces.insert(faces.end(),p.sample.begin(),p.sample.end());
        walls.insert(walls.end(),p.heldWall.begin(),p.heldWall.end());walls.insert(walls.end(),p.sampleWall.begin(),p.sampleWall.end());}
    Device<chmt::CoupledSurfaceDownload> df(faces);Device<chmt::CoupledWallDownload> dw(walls);
    std::vector<Input> inputs;std::size_t f=0,w=0;
    for(const auto& p:pairs){Input in;in.tractionScale=p.tractionScale;
        in.held={df.p?df.p+f:nullptr,dw.p?dw.p+w:nullptr,p.held.size(),p.heldWall.size()};f+=p.held.size();w+=p.heldWall.size();
        in.sample={df.p?df.p+f:nullptr,dw.p?dw.p+w:nullptr,p.sample.size(),p.sampleWall.size()};f+=p.sample.size();w+=p.sampleWall.size();inputs.push_back(in);}
    Device<Input> input(inputs);std::vector<ugkwp::GasDriveControls> controls;
    std::vector<chmt::ResidentMaterialDriveInterval> times;
    std::vector<ugkwp::GasDriveSelection> result(tests.size());
    for(std::size_t i=0;i<tests.size();++i){controls.push_back(tests[i].controls);
        times.push_back(tests[i].intervals[0]);times.push_back(tests[i].intervals[1]);result[i].end=ugkp::CompensatedTime(-17);}
    Device<ugkwp::GasDriveControls> dc(controls);Device<chmt::ResidentMaterialDriveInterval> di(times);
    Device<ugkwp::GasDriveSelection> dr(result);Device<int> scratch(std::vector<int>(2*tests.size()));check(cudaDeviceSynchronize());
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(std::size_t i=0;i<tests.size();++i){const TestHistory history{input.p+tests[i].pair,di.p+2*i,tests[i].frames};
        check(ugkwp::selectGasDriveInterval<chmt::Ns>(history,dc.p+i,dr.p+i,scratch.p+2*i,stream));}
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("drive selection graph contains a non-kernel node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));for(int replay=0;replay<3;++replay)check(cudaGraphLaunch(executable,stream));
    check(cudaStreamSynchronize(stream));if(!result.empty())check(cudaMemcpy(result.data(),dr.p,result.size()*sizeof(*dr.p),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}

}
