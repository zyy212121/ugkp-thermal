#include "sweep_line_search_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {T* p=nullptr;explicit Device(const std::vector<T>& a){if(!a.empty()){check(cudaMalloc(&p,a.size()*sizeof(T)));check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}}~Device(){if(p)cudaFree(p);}};
}
std::vector<SweepSearchProbe> probeSweepSearch(const g::SweepResidualDefinition& d,const std::vector<g::Point>& origin,const std::vector<g::Point>& correction,double metric,int limit){
    Device<g::Point> old(d.old),start(origin),delta(correction);Device<double> target(d.targets),scales(d.scales),ceiling(d.faceCeiling),tolerance(d.planeTolerance);Device<g::PlanarConstraint> planes(d.planar);
    g::ResidentSweepResidual residual(d.topology,d.old.size(),d.faces,d.planar.size(),d.emptyAxis,d.emptyMate);
    g::ResidentSweepLineSearch search(d.old.size(),residual);
    g::SweepResidualProblem problem{old.p,nullptr,target.p,scales.p,ceiling.p,tolerance.p,planes.p,d.absoluteTolerance,d.relativeTolerance,d.rankTolerance};
    Device<g::SweepLineSearchProblem> input(std::vector<g::SweepLineSearchProblem>{{problem,start.p,delta.p,metric,limit}});
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaGraphCreate(&graph,0));
    search.appendTo(graph,nullptr,input.p);check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    if(const char* path=std::getenv("UGKP_SWEEP_SEARCH_DOT"))check(cudaGraphDebugDotPrint(graph,path,cudaGraphDebugDotFlagsVerbose));
    for(int repeat=0;repeat<3;++repeat)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    std::vector<SweepSearchProbe> out;
    for(int batch=0;batch<1077;++batch){SweepSearchProbe result;const auto view=search.view();
        check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
        if(result.status.candidate){result.points.resize(origin.size());if(!origin.empty())check(cudaMemcpy(result.points.data(),view.points,origin.size()*sizeof(g::Point),cudaMemcpyDeviceToHost));}
        out.push_back(result);if(!result.status.candidate)break;
        search.resume(stream);check(cudaStreamSynchronize(stream));
        if(batch==1076)throw std::runtime_error("line search failed to exhaust native alpha sequence");
    }
    search.begin(input.p,stream);check(cudaStreamSynchronize(stream));g::SweepLineSearchStatus reset;
    check(cudaMemcpy(&reset,search.view().status,sizeof(reset),cudaMemcpyDeviceToHost));
    if(reset.nextBacktrack!=out.front().status.nextBacktrack||reset.candidate!=out.front().status.candidate||reset.trials!=out.front().status.trials)throw std::runtime_error("line search restart differs from composed graph");
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return out;
}
