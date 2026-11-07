#include "interface_displacement_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& data){if(!data.empty()){check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};
struct Record {double before=-327,depth=0,after=683,extra=-971;};
struct Field {
    Device<double> contiguous;Device<Record> records;g::InterfaceThicknessField view;
    static std::vector<Record> pack(const std::vector<double>& values,bool strided){std::vector<Record> r;if(strided)for(double x:values)r.push_back({-327,x,683,-971});return r;}
    Field(const std::vector<double>& values,bool strided):contiguous(strided?std::vector<double>{}:values),records(pack(values,strided)){
        view={strided&&records.p?&records.p[0].depth:contiguous.p,strided?sizeof(Record):sizeof(double)};}
};
__global__ void selectInterfaceInput(const g::InterfacePointDisplacementInput* seed,g::InterfacePointDisplacementInput* output){*output=*seed;}
}
InterfaceDisplacementProbe probeInterfaceDisplacement(const g::Topology& solid,int gasCount,
    const std::vector<int>& faces,const std::vector<int>& mapping,const std::vector<g::Point>& oldSolid,
    const std::vector<g::Point>& endpoint,const std::vector<g::Point>& areas,const std::vector<g::Point>& oldGas,
    const std::vector<double>& depth,const std::vector<double>& previousFilm,const std::vector<double>& film,bool strided){
    g::ResidentInterfacePointDisplacement owner(solid,oldSolid.size(),gasCount,faces,mapping);
    Device<g::Point> old(oldSolid),end(endpoint),area(areas),gas(oldGas);Field d(depth,strided),before(previousFilm,strided),after(film,strided);
    Device<g::InterfacePointDisplacementInput> seed(std::vector<g::InterfacePointDisplacementInput>{{old.p,end.p,area.p,gas.p,d.view,before.view,after.view}}),
        input(std::vector<g::InterfacePointDisplacementInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));selectInterfaceInput<<<1,1,0,stream>>>(seed.p,input.p);owner.evaluate(input.p,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("interface displacement graph has host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    InterfaceDisplacementProbe result;result.displacement.resize(gasCount);const auto view=owner.view();
    check(cudaMemcpy(&result.changed,view.changed,sizeof(int),cudaMemcpyDeviceToHost));
    if(gasCount)check(cudaMemcpy(result.displacement.data(),view.displacement,result.displacement.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
