#include "interface_stage_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace g=ugkp::gpu::geometry;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& data){if(!data.empty()){check(cudaMalloc(&p,data.size()*sizeof(T)));check(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice));}}
    ~Device(){if(p)cudaFree(p);}
};
__global__ void selectInterfaceMotion(const g::InterfaceGasStageInput* seed,g::InterfaceGasStageInput* input){*input=*seed;}
}
g::InterfaceGasStageResult probeInterfaceStage(g::InterfaceGasStage& owner,const g::InterfaceGasStageDefinition& definition,
    const std::vector<g::Point>& endpoint,const std::vector<g::Point>& areas,g::StageGeometryControls controls){
    const auto& d=definition.motion;
    Device<g::Point> old(d.oldSolid),end(endpoint),area(areas),gas(d.gas.points),reference(d.gas.reference);
    Device<double> depth(d.pointThickness),previous(d.previousFilmThickness),film(d.filmThickness);
    Device<g::InterfaceGasStageInput> seed(std::vector<g::InterfaceGasStageInput>{{{
        {old.p,end.p,area.p,gas.p,{depth.p,sizeof(double)},{previous.p,sizeof(double)},{film.p,sizeof(double)}},reference.p,d.gas.geometricTolerance},controls}}),
        input(std::vector<g::InterfaceGasStageInput>(1));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;cudaGraph_t child;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaGraphCreate(&graph,0));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));selectInterfaceMotion<<<1,1,0,stream>>>(seed.p,input.p);check(cudaStreamEndCapture(stream,&child));
    cudaGraphNode_t first;check(cudaGraphAddChildGraphNode(&first,graph,nullptr,0,child));check(cudaGraphDestroy(child));
    owner.appendTo(graph,first,input.p);check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    static bool exported=false;if(!exported){check(cudaGraphDebugDotPrint(graph,"interface-stage-graph.dot",cudaGraphDebugDotFlagsVerbose));exported=true;}
    for(int replay=0;replay<2;++replay)check(cudaGraphLaunch(executable,stream));
    // Also exercise the standalone graph after the externally composed graph.
    owner.prepare(input.p,stream);check(cudaStreamSynchronize(stream));
    const auto view=owner.view();g::InterfaceGasStageResult result;
    check(cudaMemcpy(&result.motion.status,view.motion.status,sizeof(result.motion.status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.status,view.geometry.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.diagnostics,view.diagnostics,sizeof(result.diagnostics),cudaMemcpyDeviceToHost));
    const auto read=[&](auto& out,const auto* source,int count){out.resize(count);if(count)check(cudaMemcpy(out.data(),source,count*sizeof(*source),cudaMemcpyDeviceToHost));};
    if(result.motion.status.failure==g::InterfaceGasMotionFailure::None&&result.status.failure==g::StageGeometryFailure::None){
        const auto v=view.geometry;read(result.motion.points,view.motion.points,d.gas.points.size());
        read(result.endpoint.faces,v.endpointGeometry.face,v.endpointGeometry.faces);read(result.endpoint.cells,v.endpointGeometry.cell,v.endpointGeometry.cells);
        read(result.endpoint.distance,v.endpointGeometry.distance,v.endpointGeometry.cells);read(result.oldVolume,v.oldVolume,v.endpointGeometry.cells);
        read(result.sweep,v.sweptVolume,v.endpointGeometry.faces);read(result.velocity,v.meshVelocity,v.endpointGeometry.faces);
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
