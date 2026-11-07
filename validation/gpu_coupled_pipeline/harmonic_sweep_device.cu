#include "gpu/HarmonicSweepCorrection.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <limits>
namespace {
namespace g=ugkp::gpu::geometry;
void checked(cudaError_t e,int line){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorString(e))+" at probe line "+std::to_string(line));}
#define check(call) checked((call),__LINE__)
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(const std::vector<T>& data){if(!data.empty()){
        check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};

}
g::HarmonicSweepResult harmonicSweepDeviceProbe(g::HarmonicSweepCorrection& owner,const g::HarmonicSweepDefinition& d,
    const std::vector<g::Point>& endpoint,const std::vector<double>& rhs,double rank,double consistency){
    Device<g::Point> old(d.points),end(endpoint);Device<double> scales(d.scales),b(rhs);
    Device<g::PlanarConstraint> planar(d.planar);Device<g::HarmonicBoundaryMode> modes(d.modes);
    const g::HarmonicSweepDeviceProblem value{old.p,end.p,scales.p,planar.p,modes.p,b.p,
        static_cast<int>(d.modes.size()),d.harmonicTolerance,rank,consistency,static_cast<int>(d.planar.size())};
    Device<g::HarmonicSweepDeviceProblem> problem(std::vector<g::HarmonicSweepDeviceProblem>{value});
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaGraphCreate(&graph,0));
    owner.appendTo(graph,nullptr,problem.p);
    if(const char* path=std::getenv("UGKP_TEST_GRAPH_DOT"))
        check(cudaGraphDebugDotPrint(graph,path,cudaGraphDebugDotFlagsVerbose));
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
    // Also exercise the standalone resident entry point with the same borrowed
    // problem, without host observations or numerical decisions between runs.
    owner.solveDevice(problem.p,stream);check(cudaStreamSynchronize(stream));
    const auto view=owner.view();g::HarmonicSweepResult out;
    check(cudaMemcpy(&out.status,view.status,sizeof(out.status),cudaMemcpyDeviceToHost));
    out.modes.resize(out.status.modeSolves);
    if(!out.modes.empty())check(cudaMemcpy(out.modes.data(),view.modes,out.modes.size()*sizeof(out.modes[0]),cudaMemcpyDeviceToHost));
    if(out.status.linearValid){out.rhs.resize(rhs.size());if(!rhs.empty())check(cudaMemcpy(out.rhs.data(),view.linear.rhs,rhs.size()*sizeof(double),cudaMemcpyDeviceToHost));}
    if(!out.status.error){out.correction.resize(d.points.size());if(!out.correction.empty())
        check(cudaMemcpy(out.correction.data(),view.correction,out.correction.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));}
    for(int count:{-2,std::numeric_limits<int>::max()}){
        auto invalid=value;invalid.planarCount=count;
        check(cudaMemcpyAsync(problem.p,&invalid,sizeof(invalid),cudaMemcpyHostToDevice,stream));owner.solveDevice(problem.p,stream);check(cudaStreamSynchronize(stream));
        g::HarmonicSweepStatus status;check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
        if(status.error!=1)throw std::runtime_error("invalid active plane count accepted");
    }
    check(cudaMemcpyAsync(problem.p,&value,sizeof(value),cudaMemcpyHostToDevice,stream));
    check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    g::HarmonicSweepStatus restored;check(cudaMemcpy(&restored,view.status,sizeof(restored),cudaMemcpyDeviceToHost));
    if(restored.error!=out.status.error||restored.rows!=out.status.rows)throw std::runtime_error("invalid layout latched across graph reuse");
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return out;
}
