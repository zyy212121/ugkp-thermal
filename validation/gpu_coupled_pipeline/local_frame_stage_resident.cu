#include "local_frame_stage_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {T* p=nullptr;explicit Device(const std::vector<T>& a){if(!a.empty()){check(cudaMalloc(&p,a.size()*sizeof(T)));check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}}~Device(){if(p)cudaFree(p);}};
}
std::vector<g::LocalFrameStageResult> probeLocalFrameStage(g::LocalFrameStageGeometry& owner,const g::LocalFrameStageDefinition& d,const std::vector<g::Point>& endpoint,g::StageGeometryControls controls){
    Device<g::Point> world(d.worldOld),local(d.localOld),end(endpoint);
    check(cudaStreamSynchronize(nullptr)); // Borrowed inputs must be ready before a nonblocking consumer stream.
    std::vector<g::LocalFrameStageResult> results;results.push_back(owner.solveBorrowed(end.p,controls));
    Device<g::LocalFrameStageInput> input(std::vector<g::LocalFrameStageInput>{{world.p,local.p,end.p,controls}});
    check(cudaStreamSynchronize(nullptr));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaGraphCreate(&graph,0));
    owner.appendTo(graph,nullptr,input.p);check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    if(const char* path=std::getenv("UGKP_LOCAL_FRAME_STAGE_DOT"))check(cudaGraphDebugDotPrint(graph,path,cudaGraphDebugDotFlagsVerbose));
    for(int repeat=0;repeat<2;++repeat)check(cudaGraphLaunch(executable,stream));owner.prepare(input.p,stream);check(cudaStreamSynchronize(stream));
    const auto view=owner.view();g::LocalFrameStageResult result;
    check(cudaMemcpy(&result.status,view.geometry.status,sizeof(result.status),cudaMemcpyDeviceToHost));check(cudaMemcpy(&result.diagnostics,view.diagnostics,sizeof(result.diagnostics),cudaMemcpyDeviceToHost));
    const auto read=[&](auto& values,const auto* source,int count){values.resize(count);if(count)check(cudaMemcpy(values.data(),source,std::size_t(count)*sizeof(*source),cudaMemcpyDeviceToHost));};
    if(result.status.failure==g::StageGeometryFailure::None){const auto v=view.geometry;
        read(result.points,view.points,endpoint.size());read(result.endpoint.faces,v.endpointGeometry.face,v.endpointGeometry.faces);read(result.endpoint.cells,v.endpointGeometry.cell,v.endpointGeometry.cells);
        read(result.endpoint.distance,v.endpointGeometry.distance,v.endpointGeometry.cells);read(result.oldVolume,v.oldVolume,v.endpointGeometry.cells);read(result.sweep,v.sweptVolume,v.endpointGeometry.faces);read(result.velocity,v.meshVelocity,v.endpointGeometry.faces);}
    results.push_back(result);results.push_back(owner.solveBorrowed(end.p,controls));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return results;
}
