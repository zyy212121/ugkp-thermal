#include "sweep_residual_probe.H"
#include "gpu/HarmonicSweepCorrection.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;
    explicit Device(const std::vector<T>& data){if(!data.empty()){check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};
__global__ void selectResidual(const g::SweepResidualProblem* seed,g::SweepResidualProblem* input){*input=*seed;}
}
SweepResidualProbe probeSweepResidual(const g::SweepResidualDefinition& d,const std::vector<g::Point>& end){
    g::ResidentSweepResidual owner(d.topology,d.old.size(),d.faces,d.planar.size(),d.emptyAxis,d.emptyMate);
    Device<g::Point> old(d.old),endpoint(end);Device<double> targets(d.targets),scales(d.scales),ceilings(d.faceCeiling),tolerance(d.planeTolerance);
    Device<g::PlanarConstraint> planes(d.planar);
    Device<g::SweepResidualProblem> seed(std::vector<g::SweepResidualProblem>{{old.p,endpoint.p,targets.p,scales.p,ceilings.p,tolerance.p,planes.p,d.absoluteTolerance,d.relativeTolerance,d.rankTolerance}}),
        input(std::vector<g::SweepResidualProblem>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));selectResidual<<<1,1,0,stream>>>(seed.p,input.p);owner.evaluate(input.p,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("residual graph has host/transfer node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    SweepResidualProbe result;const auto view=owner.view();check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    result.sweeps.resize(d.topology.faceOffsets.size()-1);result.residual.resize(d.faces.size()+3*d.planar.size());result.rhs.resize(result.residual.size());
    if(!result.sweeps.empty())check(cudaMemcpy(result.sweeps.data(),view.sweptFaces,result.sweeps.size()*sizeof(g::SweepResult),cudaMemcpyDeviceToHost));
    if(!result.residual.empty()){check(cudaMemcpy(result.residual.data(),view.residual,result.residual.size()*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(result.rhs.data(),view.rhs,result.rhs.size()*sizeof(double),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
g::HarmonicSweepResult harmonicSweepDeviceProbe(g::HarmonicSweepCorrection& owner,const g::HarmonicSweepDefinition&,
    const std::vector<g::Point>& end,const std::vector<double>& rhs,double rank,double consistency){
    Device<g::Point> endpoint(end);Device<double> b(rhs),tolerance(std::vector<double>{consistency});
    return owner.solveBorrowed(endpoint.p,b.p,tolerance.p,rank);
}

namespace {
__global__ void sweepRatios(int count,const double* residual,const double* tolerance,double* output){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=g::sweepResidualRatio(residual[i],tolerance[i]);}
}
std::vector<double> probeSweepRatios(const std::vector<double>& residual,const std::vector<double>& tolerance){
    Device<double> r(residual),t(tolerance),out(std::vector<double>(residual.size()));
    if(!residual.empty()){sweepRatios<<<(residual.size()-1)/128+1,128>>>(residual.size(),r.p,t.p,out.p);check(cudaGetLastError());}
    std::vector<double> result(residual.size());if(!result.empty())check(cudaMemcpy(result.data(),out.p,result.size()*sizeof(double),cudaMemcpyDeviceToHost));return result;
}
